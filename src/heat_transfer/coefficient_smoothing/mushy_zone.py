import math

import numpy as np
from numba import njit
from scipy.optimize import root_scalar
from scipy.special import erf

from src.parameters.config import ExperimentConfig
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


def find_optimal_delta(
    u_n_non,
    sf,
    time,
    solver,
    cfg: ExperimentConfig,
    delta_min,
    delta_max,
    tol=1e-3,
):
    conv_scale = cfg.l / cfg.v
    delta_u = cfg.delta_u
    u_ref = cfg.u_ref
    u_pt = cfg.material_props.u_pt
    dt = cfg.geometry.dt
    h_x = cfg.geometry.dx
    h_y = cfg.geometry.dy
    latent_vol = cfg.material_props.volumetric_latent_heat
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

    result = root_scalar(
        residual, method="brentq", bracket=[delta_min, delta_max], xtol=tol, rtol=tol
    )

    if not result.converged:
        return None

    return result.root


def get_delta(
    u_n_non,
    sf,
    time,
    solver,
    cfg: ExperimentConfig,
    delta_min,
    delta_max,
    tol=1e-3,
) -> float:
    absolute_temp = u_n_non * cfg.delta_u + cfg.u_ref
    max_delta = get_mushy_zone_temperature_range(
        u=absolute_temp,
        u_pt=cfg.material_props.u_pt,
    )
    try:
        delta = find_optimal_delta(
            u_n_non=u_n_non,
            sf=sf,
            time=time,
            solver=solver,
            cfg=cfg,
            delta_min=delta_min,
            delta_max=delta_max,
            tol=tol,
        )
        if delta is not None:
            if abs(max_delta - delta) / max_delta < 0.5:
                return delta
    except ValueError as e:
        pass
    return max_delta


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
