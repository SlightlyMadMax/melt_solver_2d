import math

import numpy as np
from numba import njit
from scipy.ndimage import (
    uniform_filter,
    median_filter,
    gaussian_filter,
)

from src.utils.array_masks import dilate_mask


@njit
def get_mushy_zone_temperature_range(
    u: np.ndarray,
    u_pt: float,
    h_x: float,
    h_y: float,
    min_delta: float = 1e-3,
    n_width: int = 4,
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
    return delta


def find_interface_cells(u, u_pt):
    """
    Identify cells where u crosses u_pt with any of the 8 neighbors.
    """
    mask = np.zeros_like(u, dtype=bool)
    diff = u - u_pt

    # 4 orthogonal neighbors
    mask[:-1, :] |= (diff[:-1, :] * diff[1:, :]) < 0  # up/down
    mask[1:, :] |= (diff[:-1, :] * diff[1:, :]) < 0
    mask[:, :-1] |= (diff[:, :-1] * diff[:, 1:]) < 0  # left/right
    mask[:, 1:] |= (diff[:, :-1] * diff[:, 1:]) < 0

    # 4 diagonal neighbors
    mask[:-1, :-1] |= (diff[:-1, :-1] * diff[1:, 1:]) < 0
    mask[1:, 1:] |= (diff[:-1, :-1] * diff[1:, 1:]) < 0
    mask[:-1, 1:] |= (diff[:-1, 1:] * diff[1:, :-1]) < 0
    mask[1:, :-1] |= (diff[:-1, 1:] * diff[1:, :-1]) < 0

    return mask


# def get_mushy_zone_temperature_range(
#     u: np.ndarray,
#     u_pt: float,
#     h_x: float,
#     h_y: float,
#     min_delta: float = 1e-3,
#     n_width: int = 5,
# ) -> np.ndarray:
#     delta = np.full_like(u, min_delta, dtype=float)
#
#     u_s = uniform_filter(u, size=5, mode="nearest")
#
#     dudx = (u_s[:, 2:] - u_s[:, :-2]) / (2 * h_x)
#     dudy = (u_s[2:, :] - u_s[:-2, :]) / (2 * h_y)
#
#     dudx = np.pad(dudx, ((0, 0), (1, 1)), mode="edge")
#     dudy = np.pad(dudy, ((1, 1), (0, 0)), mode="edge")
#
#     dudx = gaussian_filter(dudx, sigma=1.0, mode="nearest")
#     dudy = gaussian_filter(dudy, sigma=1.0, mode="nearest")
#
#     mag_grad = np.hypot(dudx, dudy) + 1e-12
#
#     interface = find_interface_cells(u, u_pt)
#     js, is_ = np.nonzero(interface)
#
#     global_delta = u.max() - u_pt
#     for j, i in zip(js, is_):
#         g = mag_grad[j, i]
#         if g < 1e-6:
#             continue
#
#         nx = dudx[j, i] / g
#         ny = dudy[j, i] / g
#
#         grad_n = nx * dudx[j, i] + ny * dudy[j, i]
#
#         grad_n = np.sign(grad_n) * min(abs(grad_n), g)
#
#         ds = np.hypot(nx * h_x, ny * h_y)
#         ds = max(ds, min(h_x, h_y))
#
#         delta_raw = abs(grad_n) * (n_width * ds)
#
#         delta[j, i] = np.clip(delta_raw, min_delta, global_delta)
#
#     delta = median_filter(delta, size=3, mode="nearest")
#     return delta


@njit
def melt_fraction_gauss(u, u0, delta):
    """Returns melt fraction ∈ [0, 1] using the Gaussian CDF."""
    return 0.5 * (1 + math.erf((u - u0) / (math.sqrt(2) * delta)))


def compute_qs(
    u_n_dim,
    u_np1_non,
    u_pt,
    conv_x,
    conv_y,
    delta_u,
    u_ref,
    latent_heat,
    delta,
    dt,
    h_x,
    h_y,
    delta_fn,
):
    """
    Returns (q_latent, q_theory) for a candidate uniform `delta`.
    - u_n_dim:    dimensional field at time n
    - u_n_non:    nondim field at time n
    - u_np1_non:  nondim field at time n+1 (from solver)
    - u_pt:       phase‐change temp (dimensional)
    - latent_heat: volumetric latent heat (J/m³)
    - delta:      mushy‐zone width (dimensional)
    - dt, h_x, h_y: time‐step and grid spacings
    - delta_fn:   kernel fn: (u_dim, u_pt, delta) -> weight
    """
    # convert solver output to dimensional
    u_np1_dim = u_np1_non * delta_u + u_ref

    ny, nx = u_n_dim.shape

    latent_field = np.zeros_like(u_n_dim)
    for j in range(ny):
        for i in range(nx):
            latent_field[j, i] = latent_heat * delta_fn(u_n_dim[j, i], u_pt, delta)

    q_dot = np.zeros_like(u_n_dim)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            a = (
                conv_x[j, i, 0] * u_n_dim[j, i + 1]
                + conv_x[j, i, 1] * u_n_dim[j, i]
                + conv_x[j, i, 2] * u_n_dim[j, i - 1]
            )
            b = (
                conv_y[j, i, 0] * u_n_dim[j + 1, i]
                + conv_y[j, i, 1] * u_n_dim[j, i]
                + conv_y[j, i, 2] * u_n_dim[j - 1, i]
            )
            convective_term = a + b

            # Latent heat release including convection
            q_dot[j, i] = latent_field[j, i] * (
                (u_np1_dim[j, i] - u_n_dim[j, i]) / dt + convective_term
            )

    # Sum latent heat over domain
    q_latent = q_dot.sum() * h_x * h_y

    solid_n = u_n_dim <= u_pt
    solid_np1 = u_np1_dim <= u_pt
    v_n = solid_n.sum() * h_x * h_y
    v_np1 = solid_np1.sum() * h_x * h_y
    q_theory = latent_heat * (v_n - v_np1)

    return q_latent, q_theory


def find_best_delta(
    u_n_dim,
    u_n_non,
    sf,
    time,
    delta_u,
    u_ref,
    solver,  # signature: (u=u_n_non, sf=sf, time=time, delta=delta) -> u_np1_non
    u_pt,
    latent_heat,
    dt,
    h_x,
    h_y,
    delta_fn,
    delta_min,
    delta_max,
    tol=1e-1,
    max_iter=100,
):
    def f(delta: float) -> float:
        u_c_non = solver.solve(u=u_n_non, sf=sf, time=time, delta=delta)
        qc, qtc = compute_qs(
            u_n_dim=u_n_dim,
            u_np1_non=u_c_non,
            u_pt=u_pt,
            conv_x=solver.solver._conv_x * 0.0635 / 0.027,
            conv_y=solver.solver._conv_y * 0.0635 / 0.027,
            delta_u=delta_u,
            u_ref=u_ref,
            latent_heat=latent_heat,
            delta=delta,
            dt=dt,
            h_x=h_x,
            h_y=h_y,
            delta_fn=delta_fn,
        )
        return qc - qtc

    a, b = delta_min, delta_max
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        print(f"Warning: f({a}) = {fa}, f({b}) = {fb} — no sign change.")
        return (a + b) / 2

    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        # print(f"Iteration {i}: delta = {c:.6f}, f = {fc:.4e}")

        if abs(fc) <= tol:
            # print(f"Converged at delta = {c:.6f}")
            return c

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    print(f"Max iterations reached. Returning midpoint: {(a + b) / 2}")
    return (a + b) / 2


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
