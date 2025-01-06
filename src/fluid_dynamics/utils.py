from typing import Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def get_indicator_function(
    u: float, u_pt_ref: float, delta_u: float, eps: float
) -> float:
    """
    Indicator function for the fictitious domain method.
    Is equal to 0 for liquid phase and 1 / eps^2 for solid phase.

    :param u: The temperature value (deviation from the reference temperature).
    :param u_pt_ref: The phase transition temperature (deviation from the reference temperature).
    :param delta_u: The characteristic temperature difference.
    :param eps: A big parameter.
    :return: The value of the indicator function at u.
    """
    if u * delta_u - u_pt_ref > 0.0:
        return 0.0
    return 1.0 / (eps * eps)


@njit
def compute_velocity_from_sf(
    sf: NDArray[np.float64],
    dx: float,
    dy: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute velocity components v_x and v_y from the stream function.
    """
    n_y, n_x = sf.shape
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy

    v_x = np.zeros_like(sf)
    v_y = np.zeros_like(sf)

    for j in range(1, n_y - 1):
        for i in range(1, n_x - 1):
            v_x[j, i] = (sf[j + 1, i] - sf[j - 1, i]) * 0.5 * inv_dy
            v_y[j, i] = -(sf[j, i + 1] - sf[j, i - 1]) * 0.5 * inv_dx

    return v_x, v_y
