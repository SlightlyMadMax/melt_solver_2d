import numpy as np

from src.core.geometry import DomainGeometry


def initialize_stream_function(geom: DomainGeometry) -> np.ndarray:
    """
    Initialize the stream function for the computational domain.

    The stream function is set to zero across the entire grid, which serves as the
    initial condition for solving fluid flow problems.

    :param geom: DomainGeometry object specifying the dimensions of the computational domain.
    :return: A 2D numpy array of shape (n_y, n_x) filled with zeros, representing
             the initial stream function values.
    """
    return np.zeros((geom.n_y, geom.n_x))


def initialize_vorticity(geom: DomainGeometry) -> np.ndarray:
    """
    Initialize the vorticity for the computational domain.

    The vorticity is set to zero across the entire grid, which serves as the initial
    condition for solving fluid dynamics problems.

    :param geom: DomainGeometry object specifying the dimensions of the computational domain.
    :return: A 2D numpy array of shape (n_y, n_x) filled with zeros, representing
             the initial vorticity values.
    """
    return np.zeros((geom.n_y, geom.n_x))
