from typing import Optional, Tuple

import numpy as np
from numba import njit

from scipy.special import erf
from numpy.typing import NDArray

from src.parameters.config import ExperimentConfig


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


def calculate_liquid_fraction(
    u: np.ndarray, u_pt: float, delta: float, result: np.ndarray
) -> None:
    diff_u = u - u_pt

    result[:, :] = 0.5 * (1.0 + erf(diff_u / (np.sqrt(2.0) * delta)))


def calculate_vorticity_from_sf(
    sf: NDArray[np.float64],
    result: NDArray[np.float64],
    cfg: ExperimentConfig,
):
    dx, dy, _ = cfg.scaled_grid_steps
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
    cfg: ExperimentConfig,
):
    dx, dy, _ = cfg.scaled_grid_steps
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
