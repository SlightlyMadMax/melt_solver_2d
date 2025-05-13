import numpy as np
from numba import njit

from src.utils.array_masks import dilate_mask
from src.utils.numerics import compute_gradient


@njit
def get_mushy_zone_temperature_range(
    u: np.ndarray,
    u_pt: float,
    h_x: float,
    h_y: float,
    alpha: float = 2.0,
    min_delta: float = 1e-3,
    max_radius: int = 3,  # grid points
) -> np.ndarray:
    n_y, n_x = u.shape
    delta = np.full_like(u, min_delta, dtype=np.float64)

    max_delta = 0.0
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            if (u[j + 1, i] - u_pt) * (u[j, i] - u_pt) <= 0.0:
                du = abs(u[j + 1, i] - u[j, i])
                max_delta = du if du > max_delta else max_delta
            if (u[j, i + 1] - u_pt) * (u[j, i] - u_pt) <= 0.0:
                du = abs(u[j, i + 1] - u[j, i])
                max_delta = du if du > max_delta else max_delta

    delta.fill(max_delta)

    # # Detect interface points
    # h_min = min(h_x, h_y)
    # boundary_points = []
    # for j in range(1, n_y - 1):
    #     for i in range(1, n_x - 1):
    #         for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    #             u1 = u[j, i]
    #             u2 = u[j + dj, i + di]
    #             if (u1 - u_pt) * (u2 - u_pt) <= 0.0:
    #                 grad = compute_gradient(u, i, j, h_x, h_y)
    #                 delta_val = max(min_delta, alpha * h_min * grad)
    #                 boundary_points.append((j, i, delta_val))
    #                 delta_T[j, i] = delta_val
    #                 break
    #
    # # Propagate locally within limited radius
    # for j in range(1, n_y - 1):
    #     for i in range(1, n_x - 1):
    #         if delta_T[j, i] != min_delta:
    #             continue  # Already a boundary point
    #
    #         min_dist2 = 1e10
    #         closest_delta = min_delta
    #         for jb, ib, delta_b in boundary_points:
    #             dj = jb - j
    #             di = ib - i
    #             if abs(dj) > max_radius or abs(di) > max_radius:
    #                 continue  # Outside search window
    #             dist2 = dj * dj + di * di
    #             if dist2 < min_dist2:
    #                 min_dist2 = dist2
    #                 closest_delta = delta_b
    #
    #         delta_T[j, i] = closest_delta

    return delta


@njit
def mark_mushy(u_dim, u_pt, delta, mushy_mask):
    n_y, n_x = u_dim.shape
    for j in range(n_y):
        for i in range(n_x):
            mushy_mask[j, i] = abs(u_dim[j, i] - u_pt) <= delta[j, i]


@njit
def get_dilated_mushy_mask(
    u_dim: np.ndarray, u_pt: float, delta: np.ndarray, extend_by: int = 1
) -> np.ndarray:
    mushy_mask = np.empty_like(u_dim, dtype=np.uint8)
    mushy_dilated = np.copy(mushy_mask)
    mark_mushy(u_dim, u_pt, delta, mushy_mask)
    dilate_mask(mushy_mask, mushy_dilated, extend_by)

    return mushy_dilated


@njit
def collect_mushy_cells(mushy_mask):
    n_y, n_x = mushy_mask.shape
    max_cells = n_y * n_x
    idx_j = np.empty(max_cells, dtype=np.int64)
    idx_i = np.empty(max_cells, dtype=np.int64)
    count = 0
    for j in range(n_y):
        for i in range(n_x):
            if mushy_mask[j, i]:
                idx_j[count] = j
                idx_i[count] = i
                count += 1
    # return only the filled portion
    return idx_j[:count], idx_i[:count]
