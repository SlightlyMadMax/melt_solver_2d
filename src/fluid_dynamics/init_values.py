from typing import Tuple

import numpy as np

from src.core.boundary_conditions import BoundaryConditions, BoundaryConditionType
from src.core.geometry import DomainGeometry


def initialize_stream_function(
    geometry: DomainGeometry, bcs: BoundaryConditions
) -> np.ndarray:
    """
    Initialize the stream function for the computational domain.

    The stream function is set to zero across the entire grid, which serves as the
    initial condition for solving fluid flow problems.

    :param geometry: DomainGeometry object specifying the dimensions of the computational domain.
    :param bcs: Boundary conditions.
    :return: A 2D numpy array of shape (n_y, n_x) filled with zeros, representing
             the initial stream function values.
    """
    sf = np.zeros((geometry.n_y, geometry.n_x))

    # apply bcs
    if bcs.left.boundary_type == BoundaryConditionType.DIRICHLET:
        sf[:, 0] = bcs.left.get_value(t=0.0)
    if bcs.top.boundary_type == BoundaryConditionType.DIRICHLET:
        sf[-1, :] = bcs.top.get_value(t=0.0)
    if bcs.right.boundary_type == BoundaryConditionType.DIRICHLET:
        sf[:, -1] = bcs.right.get_value(t=0.0)
    if bcs.bottom.boundary_type == BoundaryConditionType.DIRICHLET:
        sf[0, :] = bcs.bottom.get_value(t=0.0)

    return sf


def initialize_vorticity(geometry: DomainGeometry) -> np.ndarray:
    """
    Initialize the vorticity for the computational domain.

    The vorticity is set to zero across the entire grid, which serves as the initial
    condition for solving fluid dynamics problems.

    :param geometry: DomainGeometry object specifying the dimensions of the computational domain.
    :return: A 2D numpy array of shape (n_y, n_x) filled with zeros, representing
             the initial vorticity values.
    """
    return np.zeros((geometry.n_y, geometry.n_x))


def initialize_velocity(geometry: DomainGeometry) -> Tuple[np.ndarray, np.ndarray]:
    return np.zeros((geometry.n_y, geometry.n_x)), np.zeros((geometry.n_y, geometry.n_x))
