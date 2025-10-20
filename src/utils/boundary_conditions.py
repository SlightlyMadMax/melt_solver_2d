import math

import numpy as np

from src.core.boundary_conditions import BoundaryCondition, BoundaryConditionType


def t_air(t: float, n: int) -> np.ndarray:
    arr = np.zeros(n)
    arr[:] = (
        275.15
        + 2.0 * math.sin(2 * math.pi * t / (24.0 * 3600.0) - math.pi / 2)
        - 273.15
    ) / 10.0
    return arr


def const_neumann_condition(n: int, value: float) -> BoundaryCondition:
    """Time-independent Neumann BC."""
    return BoundaryCondition(
        boundary_type=BoundaryConditionType.NEUMANN,
        n=n,
        flux_func=lambda t, n: np.full(n, value),
    )


def const_dirichlet_condition(n: int, value: float) -> BoundaryCondition:
    """Time-independent Dirichlet BC."""
    return BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=n,
        value_func=lambda t, n: np.full(n, value),
    )
