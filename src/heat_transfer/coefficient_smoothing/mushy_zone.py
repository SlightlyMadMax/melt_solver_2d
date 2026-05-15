import numpy as np
from numba import njit, int32, float64


@njit
def detect_interface_nodes_indices(u, u_pt):
    ny, nx = u.shape
    cnt = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if (
                (u[j, i] - u_pt) * (u[j, i + 1] - u_pt) <= 0.0
                or (u[j, i] - u_pt) * (u[j, i - 1] - u_pt) <= 0.0
                or (u[j, i] - u_pt) * (u[j + 1, i] - u_pt) <= 0.0
                or (u[j, i] - u_pt) * (u[j - 1, i] - u_pt) <= 0.0
            ):
                cnt += 1

    iface_j = np.empty(cnt, dtype=int32)
    iface_i = np.empty(cnt, dtype=int32)
    idx = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if (
                (u[j, i] - u_pt) * (u[j, i + 1] - u_pt) <= 0.0
                or (u[j, i] - u_pt) * (u[j, i - 1] - u_pt) <= 0.0
                or (u[j, i] - u_pt) * (u[j + 1, i] - u_pt) <= 0.0
                or (u[j, i] - u_pt) * (u[j - 1, i] - u_pt) <= 0.0
            ):
                iface_j[idx] = j
                iface_i[idx] = i
                idx += 1
    return iface_j, iface_i


@njit
def compute_interface_position_in_cell(u, u_pt, j, i):
    """
    For node (j,i) that is flagged as interface, find all edge crossings between
    u[j,i] and its 4-neighbors and return the average crossing point (x_avg, y_avg)
    in index-space (x = col index, y = row index). If no crossing found, returns
    the cell center (i, j).
    """
    t_center = u[j, i]
    sum_x = 0.0
    sum_y = 0.0
    count = 0

    # neighbor (i+1, j): horizontal edge to the right
    t_nb = u[j, i + 1]
    if (t_center - u_pt) * (t_nb - u_pt) <= 0.0 and t_nb != t_center:
        frac = (u_pt - t_center) / (t_nb - t_center)  # in [0,1] normally
        x = i + frac
        y = j
        sum_x += x
        sum_y += y
        count += 1

    # neighbor (i-1, j): horizontal edge to the left
    t_nb = u[j, i - 1]
    if (t_center - u_pt) * (t_nb - u_pt) <= 0.0 and t_nb != t_center:
        frac = (u_pt - t_center) / (t_nb - t_center)
        x = i - frac
        y = j
        sum_x += x
        sum_y += y
        count += 1

    # neighbor (i, j+1): vertical edge downward (row +1)
    t_nb = u[j + 1, i]
    if (t_center - u_pt) * (t_nb - u_pt) <= 0.0 and t_nb != t_center:
        frac = (u_pt - t_center) / (t_nb - t_center)
        x = i
        y = j + frac
        sum_x += x
        sum_y += y
        count += 1

    # neighbor (i, j-1): vertical edge upward (row -1)
    t_nb = u[j - 1, i]
    if (t_center - u_pt) * (t_nb - u_pt) <= 0.0 and t_nb != t_center:
        frac = (u_pt - t_center) / (t_nb - t_center)
        x = i
        y = j - frac
        sum_x += x
        sum_y += y
        count += 1

    if count == 0:
        return float(i), float(j)
    else:
        return sum_x / count, sum_y / count


@njit
def compute_smoothed_gradients(u, h_x, h_y):
    ny, nx = u.shape
    gx = np.zeros((ny, nx), dtype=float64)  # dT/dx (physical)
    gy = np.zeros((ny, nx), dtype=float64)  # dT/dy (physical)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            gx[j, i] = (u[j, i + 1] - u[j, i - 1]) / (2.0 * h_x)
            gy[j, i] = (u[j + 1, i] - u[j - 1, i]) / (2.0 * h_y)

    # 3x3 average smoothing of gradient components
    gx_s = np.zeros((ny, nx), dtype=float64)
    gy_s = np.zeros((ny, nx), dtype=float64)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            sum_gx = 0.0
            sum_gy = 0.0
            for jj in range(j - 1, j + 2):
                for ii in range(i - 1, i + 2):
                    sum_gx += gx[jj, ii]
                    sum_gy += gy[jj, ii]
            gx_s[j, i] = sum_gx / 9.0
            gy_s[j, i] = sum_gy / 9.0

    return gx_s, gy_s


@njit
def normalize_components(gx, gy):
    ny, nx = gx.shape
    nx_comp = np.zeros((ny, nx), dtype=float64)
    ny_comp = np.zeros((ny, nx), dtype=float64)
    for j in range(ny):
        for i in range(nx):
            mag = (gx[j, i] * gx[j, i] + gy[j, i] * gy[j, i]) ** 0.5
            if mag > 1e-14:
                nx_comp[j, i] = gx[j, i] / mag
                ny_comp[j, i] = gy[j, i] / mag
            else:
                nx_comp[j, i] = 0.0
                ny_comp[j, i] = 0.0
    return nx_comp, ny_comp


# @njit
# def get_mushy_zone_temperature_range(u, u_pt, n_nodes, h_x, h_y, min_delta, max_delta):
#     """
#     Returns a global Δ (temperature difference) so mushy zone spans ~n_nodes in each phase.
#     - Uses nearest-neighbor sampling at x ± s * n, where s = n_nodes * max(h_x,h_y) (physical)
#     - Converts physical offset to index offset: i +/- (s * n_x) / h_x, j +/- (s * n_y) / h_y
#     """
#     ny, nx = u.shape
#     iface_j, iface_i = detect_interface_nodes_indices(u, u_pt)
#     n_iface = iface_j.shape[0]
#     if n_iface == 0:
#         return 0.0
#
#     gx_s, gy_s = compute_smoothed_gradients(u, h_x, h_y)
#     nx_arr, ny_arr = normalize_components(gx_s, gy_s)
#
#     # physical sampling distance
#     s = (n_nodes - 0.5) * max(h_x, h_y)
#     if s < 0.0:
#         s = 0.0
#
#     global_delta = 0.0
#     for k in range(n_iface):
#         j = iface_j[k]
#         i = iface_i[k]
#
#         # compute interface point inside the cell (x_int, y_int) in index units
#         x_int, y_int = compute_interface_position_in_cell(u, u_pt, j, i)
#
#         # normal components (physical unit)
#         n_x = nx_arr[j, i]
#         n_y = ny_arr[j, i]
#
#         # fallback to axis-aligned if gradient too small
#         if n_x == 0.0 and n_y == 0.0:
#             gx_center = 0.0
#             gy_center = 0.0
#             if 1 <= i < nx - 1:
#                 gx_center = (u[j, i + 1] - u[j, i - 1]) / (2.0 * h_x)
#             if 1 <= j < ny - 1:
#                 gy_center = (u[j + 1, i] - u[j - 1, i]) / (2.0 * h_y)
#             if abs(gx_center) >= abs(gy_center):
#                 n_x = 1.0
#                 n_y = 0.0
#             else:
#                 n_x = 0.0
#                 n_y = 1.0
#
#         # convert s*n to index offsets: di = (s * n_x) / h_x ; dj = (s * n_y) / h_y
#         i_plus_f = x_int + (s * n_x) / h_x
#         j_plus_f = y_int + (s * n_y) / h_y
#         i_minus_f = x_int - (s * n_x) / h_x
#         j_minus_f = y_int - (s * n_y) / h_y
#
#         # nearest neighbor indices
#         ip = int(np.rint(i_plus_f))
#         jp = int(np.rint(j_plus_f))
#         im = int(np.rint(i_minus_f))
#         jm = int(np.rint(j_minus_f))
#
#         # clamp
#         if ip < 0:
#             ip = 0
#         if ip > nx - 1:
#             ip = nx - 1
#         if jp < 0:
#             jp = 0
#         if jp > ny - 1:
#             jp = ny - 1
#         if im < 0:
#             im = 0
#         if im > nx - 1:
#             im = nx - 1
#         if jm < 0:
#             jm = 0
#         if jm > ny - 1:
#             jm = ny - 1
#
#         u_plus = u[jp, ip]
#         u_minus = u[jm, im]
#
#         local_delta = abs(u_pt - u_plus)
#         d2 = abs(u_pt - u_minus)
#         if d2 > local_delta:
#             local_delta = d2
#
#         if local_delta > global_delta:
#             global_delta = local_delta
#
#     return max(min(global_delta, max_delta), min_delta)


@njit
def get_mushy_zone_temperature_range(
    u: np.ndarray, u_pt: float, n_nodes: int = 1
) -> float:
    n_y, n_x = u.shape

    max_delta = 0.0
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            if (u[j + 1, i] - u_pt) * (u[j, i] - u_pt) <= 0.0:
                left_index = max(0, j - (n_nodes - 1))
                right_index = min(n_y - 1, j + n_nodes)

                du_left = abs(u[left_index, i] - u_pt)
                max_delta = du_left if du_left > max_delta else max_delta

                du_right = abs(u[right_index, i] - u_pt)
                max_delta = du_right if du_right > max_delta else max_delta
            if (u[j, i + 1] - u_pt) * (u[j, i] - u_pt) <= 0.0:
                left_index = max(0, i - (n_nodes - 1))
                right_index = min(n_x - 1, i + n_nodes)

                du_left = abs(u[j, left_index] - u_pt)
                max_delta = du_left if du_left > max_delta else max_delta

                du_right = abs(u[j, right_index] - u_pt)
                max_delta = du_right if du_right > max_delta else max_delta

    return max_delta
