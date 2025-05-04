import numpy as np
from numba import njit
from numpy.typing import NDArray

from src.utils.numerics import compute_gradient


# @njit
# def get_mushy_zone_width(
#     u: NDArray[np.float64],
#     u_pt: float,
# ) -> float:
#     """
#     Find the smoothing parameter for both axes.
#
#     :param u: A 2D array of dimensional temperatures at the current time layer.
#     :param u_pt: Dimensional phase transition temperature.
#     :return: The maximum temperature interval containing the phase transition boundary.
#     """
#     n_y, n_x = u.shape
#     max_delta = 0.0
#     for i in range(n_x - 1):
#         for j in range(n_y - 1):
#             if (u[j + 1, i] - u_pt) * (u[j, i] - u_pt) <= 0.0:
#                 delta_u = abs(u[j + 1, i] - u[j, i])
#                 max_delta = delta_u if delta_u > max_delta else max_delta
#                 # break
#             if (u[j, i + 1] - u_pt) * (u[j, i] - u_pt) <= 0.0:
#                 delta_u = abs(u[j, i + 1] - u[j, i])
#                 max_delta = delta_u if delta_u > max_delta else max_delta
#                 # break
#     return max_delta


@njit
def get_mushy_zone_width(
    u: np.ndarray,
    u_pt: float,
    h_x: float,
    h_y: float,
    alpha: float = 2.0,
    min_delta: float = 1e-3,
    max_radius: int = 5,  # grid points
) -> np.ndarray:
    n_y, n_x = u.shape
    delta_T = np.full_like(u, min_delta)
    h_min = min(h_x, h_y)

    # Detect interface points
    boundary_points = []
    for j in range(1, n_y - 1):
        for i in range(1, n_x - 1):
            for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                u1 = u[j, i]
                u2 = u[j + dj, i + di]
                if (u1 - u_pt) * (u2 - u_pt) <= 0.0:
                    grad = compute_gradient(u, i, j, h_x, h_y)
                    delta_val = max(min_delta, alpha * h_min * grad)
                    boundary_points.append((j, i, delta_val))
                    delta_T[j, i] = delta_val
                    break

    # Propagate locally within limited radius
    for j in range(1, n_y - 1):
        for i in range(1, n_x - 1):
            if delta_T[j, i] != min_delta:
                continue  # Already a boundary point

            min_dist2 = 1e10
            closest_delta = min_delta
            for jb, ib, delta_b in boundary_points:
                dj = jb - j
                di = ib - i
                if abs(dj) > max_radius or abs(di) > max_radius:
                    continue  # Outside search window
                dist2 = dj * dj + di * di
                if dist2 < min_dist2:
                    min_dist2 = dist2
                    closest_delta = delta_b

            delta_T[j, i] = closest_delta

    return delta_T
