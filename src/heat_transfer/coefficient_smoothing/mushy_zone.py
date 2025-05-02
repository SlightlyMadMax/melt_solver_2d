import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit
def get_mushy_zone_width(
    u: NDArray[np.float64],
    u_pt: float,
) -> float:
    """
    Find the smoothing parameter for both axes.

    :param u: A 2D array of dimensional temperatures at the current time layer.
    :param u_pt: Dimensional phase transition temperature.
    :return: The maximum temperature interval containing the phase transition boundary.
    """
    n_y, n_x = u.shape
    delta = 0.0
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            if (u[j + 1, i] - u_pt) * (u[j, i] - u_pt) <= 0.0:
                temp = abs(u[j + 1, i] - u[j, i])
                delta = temp if temp > delta else delta
                # break
            if (u[j, i + 1] - u_pt) * (u[j, i] - u_pt) <= 0.0:
                temp = abs(u[j, i + 1] - u[j, i])
                delta = temp if temp > delta else delta
                # break
    return delta
