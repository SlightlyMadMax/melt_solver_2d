import math

import numpy as np

from src.core.boundary_conditions import BoundaryCondition, BoundaryConditionType


def air_temp(
    n: int,
    time: float,
    u_base: float,
    u_ref: float,
    u_amp: float,
    delta_u: float,
) -> np.ndarray:
    arr = np.zeros(n)
    current_temp = u_base + u_amp * math.sin(2 * math.pi * time / (24.0 * 3600.0) - math.pi / 2)
    arr[:] = (current_temp - u_ref) / delta_u  # nondimensionalized
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


def linear_dirichlet_ramp(
    n: int,
    start_value: float,
    end_value: float,
    duration: float,
) -> BoundaryCondition:
    """
    Time-dependent Dirichlet BC where the value changes linearly
    from start_value to end_value over 'duration' seconds.

    After 'duration', the value stays equal to end_value.
    """

    if duration <= 0:
        raise ValueError("duration must be positive")

    def value_func(t: float, n: int) -> np.ndarray:
        if t <= 0.0:
            value = start_value
        elif t >= duration:
            value = end_value
        else:
            alpha = t / duration
            value = start_value + alpha * (end_value - start_value)

        return np.full(n, value)

    return BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=n,
        value_func=value_func,
    )
