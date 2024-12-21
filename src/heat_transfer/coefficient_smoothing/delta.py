import numba
import numpy as np
from numpy.typing import NDArray


@numba.jit(nopython=True)
def get_max_delta(u: NDArray[np.float64], u_pt_ref: float) -> float:
    """
    Find the smoothing parameter for both axes.

    :param u: A 2D array of temperatures at the current time layer.
    :param u_pt_ref: The phase transition heat_transfer (deviation from the reference heat_transfer).
    :return: The maximum heat_transfer interval containing the phase transition boundary.
    """
    n_y, n_x = u.shape
    delta = 0.0

    for i in range(n_x - 1):
        for j in range(n_y - 1):
            if (u[j + 1, i] - u_pt_ref) * (u[j, i] - u_pt_ref) < 0.0:
                temp = abs(u[j + 1, i] - u[j, i])
                delta = temp if temp > delta else delta
                break
            if (u[j, i + 1] - u_pt_ref) * (u[j, i] - u_pt_ref) < 0.0:
                temp = abs(u[j, i + 1] - u[j, i])
                delta = temp if temp > delta else delta
                break
    return delta
