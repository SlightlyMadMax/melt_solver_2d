from typing import Optional, Tuple

import numpy as np
from numba import njit

from scipy.special import erf
from numpy.typing import NDArray

from src.core.constants import ABS_ZERO


class VorticityBCMixin:
    @staticmethod
    def calculate_boundary_conditions(
        sf: NDArray[np.float64],
        top_bc: NDArray[np.float64],
        right_bc: NDArray[np.float64],
        bottom_bc: NDArray[np.float64],
        left_bc: NDArray[np.float64],
        order: int,
        dx: float,
        dy: float,
    ) -> None:
        dx2_2_inv = 2.0 / dx**2
        dy2_2_inv = 2.0 / dy**2
        inv_2dx2 = 1.0 / (2.0 * dx**2)
        inv_2dy2 = 1.0 / (2.0 * dy**2)

        if order == 1:
            top_bc[:] = -dy2_2_inv * sf[-2, :]
            right_bc[:] = -dx2_2_inv * sf[:, -2]
            bottom_bc[:] = -dy2_2_inv * sf[1, :]
            left_bc[:] = -dx2_2_inv * sf[:, 1]
        elif order == 2:
            top_bc[:] = inv_2dy2 * (sf[-3, :] - 8.0 * sf[-2, :])
            right_bc[:] = inv_2dx2 * (sf[:, -3] - 8.0 * sf[:, -2])
            bottom_bc[:] = inv_2dy2 * (sf[2, :] - 8.0 * sf[1, :])
            left_bc[:] = inv_2dx2 * (sf[:, 2] - 8.0 * sf[:, 1])
        else:
            raise NotImplementedError


def calculate_penalty_term(
    u: NDArray[np.float64],
    u_pt: float,
    eps: float,
    result: NDArray[np.float64],
    delta: float,
) -> None:
    """
    Penalty term for the fictitious domain method.
    Is equal to 0 for liquid phase and 1 / eps^2 for solid phase.

    :param u: The dimensional temperature value.
    :param u_pt: The phase transition temperature.
    :param eps: A small parameter.
    :param result: A ndarray for storing the calculated penalty term values.
    :return: None.
    """
    result[:, :] = 0.0

    inv_eps2 = 1.0 / (eps * eps)
    interior = (slice(1, -1), slice(1, -1))

    diff_u = u[interior] - u_pt
    # delta_inner = delta[interior]

    # --- Variant 1: sharp step ----------------------
    # mask = diff_u < 0.0
    # result_interior = np.zeros_like(diff_u)
    # result_interior[mask] = inv_eps2
    # result[interior] = result_interior

    # --- Variant 2: error‐function form -------------------
    result[interior] = 0.5 * inv_eps2 * (1.0 - erf(diff_u / (np.sqrt(2.0) * delta)))

    # --- Variant 3: hyperbolic‐tangent form ---------------
    # result[interior] = (
    #     0.5
    #     * inv_eps2
    #     * (
    #         1.0
    #         - np.tanh(
    #             3.0 * diff_u / np.sqrt(delta_inner * delta_inner - diff_u * diff_u)
    #         )
    #     )
    # )

    # --- Variant 4: exponential form (one-sided smoothing) ----------------------
    # exp_term = np.exp((delta - diff_u) / delta)
    # temp = inv_eps2 * 0.5 * (2.0 + exp_term / (0.5 - exp_term))
    # result[interior] = np.where(diff_u < 0, temp, 0.0)


def calculate_vorticity_from_sf(
    sf: NDArray[np.float64],
    result: NDArray[np.float64],
    dx: float,
    dy: float,
):
    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2

    result[0, :] = -2.0 * inv_dy2 * sf[1, :]
    result[-1, :] = -2.0 * inv_dy2 * sf[-2, :]
    result[:, 0] = -2.0 * inv_dx2 * sf[:, 1]
    result[:, -1] = -2.0 * inv_dx2 * sf[:, -2]

    result[1:-1, 1:-1] = -(
        (sf[2:, 1:-1] - 2 * sf[1:-1, 1:-1] + sf[:-2, 1:-1]) * inv_dy2
        + (sf[1:-1, 2:] - 2 * sf[1:-1, 1:-1] + sf[1:-1, :-2]) * inv_dx2
    )


def calculate_velocity_from_sf(
    sf: NDArray[np.float64],
    v_x: NDArray[np.float64],
    v_y: NDArray[np.float64],
    dx: float,
    dy: float,
):
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)

    v_x[:, :] = 0.0
    v_y[:, :] = 0.0

    v_x[1:-1, 1:-1] = inv_2dy * (sf[2:, 1:-1] - sf[:-2, 1:-1])
    v_y[1:-1, 1:-1] = -inv_2dx * (sf[1:-1, 2:] - sf[1:-1, :-2])


@njit
def check_divergence(vx, vy, dx, dy):
    ny, nx = vx.shape
    div = np.zeros_like(vx)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            div[j, i] = (vx[j, i + 1] - vx[j, i - 1]) / (2 * dx) + (
                vy[j + 1, i] - vy[j - 1, i]
            ) / (2 * dy)

    max_div = np.max(np.abs(div))
    l1_div = np.sum(np.abs(div)) * dx * dy
    net_div = np.sum(div) * dx * dy
    return max_div, l1_div, net_div


def max_sf_in_solid_phase(
    sf: NDArray[np.float64], u: NDArray[np.float64], u_pt: float
) -> Tuple[Optional[float], Tuple[Optional[int], Optional[int]]]:
    mask = u <= u_pt
    if not np.any(mask):
        return None, (None, None)

    flat_idxs = np.nonzero(mask.ravel())[0]
    values = np.abs(sf).ravel()[flat_idxs]
    sub_arg = int(np.argmax(values))
    flat_idx = int(flat_idxs[sub_arg])
    pos = np.unravel_index(flat_idx, sf.shape)

    max_val = float(values[sub_arg])
    return max_val, (int(pos[0]), int(pos[1]))


def max_speed_in_solid_phase(
    v_x: NDArray[np.float64],
    v_y: NDArray[np.float64],
    u: NDArray[np.float64],
    u_pt: float,
) -> Tuple[Optional[float], Tuple[Optional[int], Optional[int]]]:
    mask = u <= u_pt
    if not np.any(mask):
        return None, (None, None)

    speed = np.sqrt(v_x**2 + v_y**2)
    flat_idxs = np.nonzero(mask.ravel())[0]
    values = speed.ravel()[flat_idxs]
    sub_arg = int(np.argmax(values))
    flat_idx = int(flat_idxs[sub_arg])
    pos = np.unravel_index(flat_idx, speed.shape)

    max_val = float(values[sub_arg])
    return max_val, (int(pos[0]), int(pos[1]))


def calculate_density_x_gradient(
    u: NDArray[np.float64],
    u_ref: float,
    delta_u: float,
    dx: float,
    beta: float,
    rho_ref: float,
) -> NDArray[np.float64]:
    u_k = u * delta_u + u_ref
    u_c = u_k + ABS_ZERO
    drho_du = (
        0.0673268037314653
        - 2 * 0.00894484552601798 * u_c
        + 3 * 8.78462866500416e-5 * u_c**2
        - 4 * 6.62139792627547e-7 * u_c**3
    )
    du_dx = np.zeros_like(u_k)
    du_dx[:, 1:-1] = (u_k[:, 2:] - u_k[:, :-2]) / (2 * dx)
    drho_dx = drho_du * du_dx
    return drho_dx / (delta_u * beta * rho_ref)
