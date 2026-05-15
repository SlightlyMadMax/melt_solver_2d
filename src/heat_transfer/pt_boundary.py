from typing import Tuple, Optional

import numpy as np
import math
from numpy.typing import NDArray

from src.core.geometry import DomainGeometry
from src.parameters.config import ExperimentConfig


def init_crevasse_boundary(
    geom: DomainGeometry, water_th: float, crev_depth: float
) -> np.ndarray:
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
    cfg: ExperimentConfig,
    u: NDArray[np.float64],
) -> Tuple[list, list]:
    """
    Find the coordinates of the phase-transition boundary.

    :param cfg: An object containing experiment parameters (geometry, material properties, etc.).
    :param u: A 2D array of temperatures at the current time layer.
    :return: 1d arrays for x and y coordinates of the phase-transition boundary interface.
    """
    x, y = [], []
    n_y, n_x = cfg.geometry.n_y, cfg.geometry.n_x
    dy, dx = cfg.geometry.dy, cfg.geometry.dx
    u_diff = u - cfg.material_props.u_pt

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


def get_pt_quadratic(
    u0: float, u1: float, u2: float, u_pt: float, y0: float, y1: float, y2: float
) -> Optional[float]:
    if np.min([u0, u1, u2]) <= u_pt <= np.max([u0, u1, u2]):
        # Solve for a, b, c in u = a*y^2 + b*y + c
        A = np.array([[y0**2, y0, 1], [y1**2, y1, 1], [y2**2, y2, 1]])
        u_vals = np.array([u0, u1, u2])
        try:
            a, b, c = np.linalg.solve(A, u_vals)
            # Solve a*y^2 + b*y + c = u_ref => a*y^2 + b*y + (c - u_ref) = 0
            coeffs = [a, b, c - u_pt]
            roots = np.roots(coeffs)
            for root in roots:
                if np.isreal(root) and y0 <= root <= y2:
                    return float(root)
        except np.linalg.LinAlgError:
            print(f"Couldn't find the interface point between {u0, u1, u2}.")
    return None
