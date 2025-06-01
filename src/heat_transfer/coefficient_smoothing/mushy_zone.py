import math

import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy.ndimage import (
    uniform_filter,
    median_filter,
    gaussian_filter,
)
from scipy.optimize import root_scalar
from scipy.special import erf

from src.core.geometry import DomainGeometry
from src.parameters.thermal import ThermalParameters
from src.utils.array_masks import dilate_mask


@njit
def get_mushy_zone_temperature_range(u: np.ndarray, u_pt: float) -> float:
    n_y, n_x = u.shape

    max_delta = 0.0
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            if (u[j + 1, i] - u_pt) * (u[j, i] - u_pt) <= 0.0:
                du = abs(u[j + 1, i] - u[j, i])
                max_delta = du if du > max_delta else max_delta
            if (u[j, i + 1] - u_pt) * (u[j, i] - u_pt) <= 0.0:
                du = abs(u[j, i + 1] - u[j, i])
                max_delta = du if du > max_delta else max_delta

    return max_delta


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


def delta_kernel(u, u_pt, delta):
    return np.exp(-0.5 * ((u - u_pt) / delta) ** 2) / (math.sqrt(2 * math.pi) * delta)


def compute_qs(
    u_n_dim, u_np1_dim, conv_x, conv_y, latent_vol, u_pt, delta, dt, h_x, h_y
):
    sqrt_2 = math.sqrt(2.0)

    # melt fractions
    phi_n = 0.5 * (1 + erf((u_n_dim - u_pt) / (sqrt_2 * delta)))
    phi_np1 = 0.5 * (1 + erf((u_np1_dim - u_pt) / (sqrt_2 * delta)))

    # theoretical latent‐heat release
    v_liq_n = phi_n.sum() * (h_x * h_y)
    v_liq_np1 = phi_np1.sum() * (h_x * h_y)
    q_theory = latent_vol * (v_liq_np1 - v_liq_n)

    k = delta_kernel(u_n_dim, u_pt, delta)
    latent_field = latent_vol * k

    u_right = np.roll(u_np1_dim, -1, axis=1)
    u_left = np.roll(u_np1_dim, 1, axis=1)
    a = (
        conv_x[:, :, 0] * u_right
        + conv_x[:, :, 1] * u_np1_dim
        + conv_x[:, :, 2] * u_left
    )

    u_up = np.roll(u_np1_dim, -1, axis=0)
    u_down = np.roll(u_np1_dim, 1, axis=0)
    b = conv_y[:, :, 0] * u_up + conv_y[:, :, 1] * u_np1_dim + conv_y[:, :, 2] * u_down

    conv_term = a + b

    dudt = (u_np1_dim - u_n_dim) / dt

    q_dot = latent_field * (dudt + conv_term)
    q_dot[0, :] = q_dot[-1, :] = q_dot[:, 0] = q_dot[:, -1] = 0.0

    q_num = q_dot.sum() * (h_x * h_y)

    return q_num, q_theory


def find_bracket(residual, d_min, d_max, max_steps=10):
    """
    Marches from d_min → d_max in max_steps increments,
    looking for the first sign change of residual.
    Returns (a,b) such that residual(a)*residual(b) ≤ 0.
    """
    fa = residual(d_min)

    for k in range(1, max_steps + 1):
        dk = d_min + (d_max - d_min) * (k / max_steps)
        fb = residual(dk)
        if fa * fb <= 0:
            # Found sign‐change bracket [previous d, current d]
            return d_min + (d_max - d_min) * ((k - 1) / max_steps), dk
        fa, d_min = fb, dk

    raise ValueError(
        f"No sign change found in [{d_min:.3g}, {d_max:.3g}] "
        f"after {max_steps} steps; residual_min={fa:.3e}"
    )


def find_optimal_delta(
    u_n_non,
    sf,
    time,
    solver,
    params: ThermalParameters,
    geometry: DomainGeometry,
    delta_min,
    delta_max,
    tol=1e-3,
    bracket_steps=10,
):
    conv_scale = params.l / params.v
    delta_u = params.delta_u
    u_ref = params.u_ref
    u_pt = params.u_pt
    dt = geometry.dt
    h_x = geometry.dx
    h_y = geometry.dy
    latent_vol = params.volumetric_latent_heat
    u_n_dim = u_n_non * delta_u + u_ref
    conv_x = solver.solver._conv_x * conv_scale
    conv_y = solver.solver._conv_y * conv_scale

    def residual(delta: float) -> float:
        u_np1_non = solver.solve(u=u_n_non, sf=sf, time=time, delta=delta)
        u_np1_dim = u_np1_non * delta_u + u_ref
        q_num, q_t = compute_qs(
            u_n_dim, u_np1_dim, conv_x, conv_y, latent_vol, u_pt, delta, dt, h_x, h_y
        )
        return q_num - q_t

    a, b = find_bracket(residual, delta_min, delta_max, max_steps=bracket_steps)

    result = root_scalar(residual, method="brentq", bracket=[a, b], xtol=tol, rtol=tol)

    if not result.converged:
        return None

    # res = []
    # deltas = np.linspace(delta_min, delta_max, 20)
    # for delta in deltas:
    #     res.append(residual(delta))
    #
    # plt.plot(deltas, res, linestyle="--", marker="o")
    # plt.grid()
    # plt.show()

    return result.root


def get_delta(
    u_n_non,
    sf,
    time,
    solver,
    params: ThermalParameters,
    geometry: DomainGeometry,
    delta_min,
    delta_max,
    tol=1e-3,
) -> float:
    try:
        delta = find_optimal_delta(
            u_n_non=u_n_non,
            sf=sf,
            time=time,
            solver=solver,
            params=params,
            geometry=geometry,
            delta_min=delta_min,
            delta_max=delta_max,
            tol=tol,
        )
        if delta is not None:
            # print(f"yooo: {delta}")
            return delta
    except ValueError as e:
        pass
    absolute_temp = u_n_non * params.delta_u + params.u_ref
    return get_mushy_zone_temperature_range(
        u=absolute_temp,
        u_pt=params.u_pt,
    )


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
