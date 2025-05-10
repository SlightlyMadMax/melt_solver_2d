from typing import Tuple

import numpy as np
import math
from numpy.typing import NDArray

from src.core.geometry import DomainGeometry


def init_crevasse_boundary(geom: DomainGeometry, water_th: float, crev_depth: float):
    """
    Initialize the position of the boundary interface for an ice crevasse filled with water.

    :param geom: An object containing the geometry information.
    :param water_th: The thickness of the layer of water covering the crevasse.
    :param crev_depth: The maximum depth of the crevasse.
    :return: A 1D array of x coordinates.
    """
    f = np.empty(geom.n_x)

    f[:] = [
        geom.height
        - water_th
        - crev_depth * math.exp(-((i * geom.dx - 0.5) ** 2) / 0.005)
        for i in range(geom.n_x)
    ]

    return f


def get_phase_trans_boundary(
    geom: DomainGeometry,
    u: NDArray[np.float64],
    u_pt: float,
) -> Tuple[list, list]:
    """
    Find the coordinates of the phase-transition boundary.

    :param geom: Object containing geometry information.
    :param u: A 2D array of temperatures at the current time layer.
    :param u_pt: The phase transition temperature.
    :return: 1d arrays for x and y coordinates of the phase-transition boundary interface.
    """
    x, y = [], []
    n_y, n_x = geom.n_y, geom.n_x
    dy, dx = geom.dy, geom.dx
    u_diff = u - u_pt

    for j in range(1, n_y - 1):
        for i in range(1, n_x - 1):
            if u_diff[j, i] * u_diff[j + 1, i] <= 0.0:
                y_0 = j * dy + (-u_diff[j, i] / (u[j + 1, i] - u[j, i])) * dy
                y.append(y_0)
                x.append(i * dx)
            if u_diff[j, i] * u_diff[j, i + 1] <= 0.0:
                x_0 = i * dx + (-u_diff[j, i] / (u[j, i + 1] - u[j, i])) * dx
                x.append(x_0)
                y.append(j * dy)

    return x, y
