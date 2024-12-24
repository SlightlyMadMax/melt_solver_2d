import numba
import numpy as np
from numpy.typing import NDArray

from src.constants import ABS_ZERO


@numba.jit(nopython=True)
def get_indicator_function(u: float, u_pt_ref: float, eps: float) -> float:
    """
    Indicator function for the fictitious domain method.
    Is equal to 1 for liquid phase and 1 / eps^2 for solid phase.

    :param u: The heat_transfer value (deviation from the reference heat_transfer).
    :param u_pt_ref: The phase transition heat_transfer (deviation from the reference heat_transfer).
    :param eps: A big parameter.
    :return: The value of the indicator function at u.
    """
    if u - u_pt_ref > 0.0:
        return 1.0
    return 1.0 / (eps * eps)


@numba.jit(nopython=True)
def calculate_velocity_field(sf: NDArray[np.float64], dx: float, dy: float):
    """
    Calculate the velocity field based on the values of the stream function using finite differences.

    :param sf: A 2D array of stream function values.
    :param dx: X-axis grid step.
    :param dy: Y-axis grid step.
    :return: v_x, v_y 2D arrays.
    """
    inv_dy = 1.0 / dy
    inv_dx = 1.0 / dx

    v_x = np.zeros_like(sf)
    v_y = np.zeros_like(sf)

    # Interior points: central difference
    v_x[1:-1, :] = 0.5 * inv_dy * (sf[2:, :] - sf[:-2, :])
    v_y[:, 1:-1] = -0.5 * inv_dx * (sf[:, 2:] - sf[:, :-2])

    # Boundary points: forward/backward difference
    v_x[0, :] = (sf[1, :] - sf[0, :]) * inv_dy
    v_x[-1, :] = (sf[-1, :] - sf[-2, :]) * inv_dy
    v_y[:, 0] = -(sf[:, 1] - sf[:, 0]) * inv_dx
    v_y[:, -1] = -(sf[:, -1] - sf[:, -2]) * inv_dx

    return v_x, v_y
