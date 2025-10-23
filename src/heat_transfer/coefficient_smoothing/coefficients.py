import math
from enum import Enum
from typing import Callable

from numba import njit


class StepScheme(Enum):
    ERF = "erf"
    HYPER = "hyper"
    LINEAR = "linear"
    CONST = "const"


class DeltaScheme(Enum):
    GAUSS = "gauss"
    HYPER = "hyper"
    PARABOLIC = "parabolic"
    BOX = "box"


@njit
def step_erf(u: float, u0: float, delta: float) -> float:
    return 0.5 * (1.0 + math.erf((u - u0) / (math.sqrt(2) * delta)))


@njit
def step_hyper(u: float, u0: float, delta: float) -> float:
    diff = u - u0
    if abs(diff) < delta:
        return 0.5 * (1.0 + math.tanh(3.0 * diff / math.sqrt(delta**2 - diff**2)))
    return 1.0 if u > u0 else 0.0


@njit
def step_linear(u: float, u0: float, delta: float) -> float:
    diff = u - u0
    if diff >= delta:
        return 1.0
    elif diff <= -delta:
        return 0.0
    return (diff + delta) / (2.0 * delta)


@njit
def step_const(u: float, u0: float, delta: float) -> float:
    diff = u - u0
    if diff >= delta:
        return 1.0
    elif diff <= -delta:
        return 0.0
    return 0.5


@njit
def delta_gauss(u: float, u0: float, delta: float) -> float:
    return math.exp(-((u - u0) ** 2) / (2 * delta**2)) / (
        math.sqrt(2 * math.pi) * delta
    )


@njit
def delta_hyper(u: float, u0: float, delta: float) -> float:
    diff = u - u0
    if abs(diff) < delta:
        return (1.5 * (delta**2 / (delta**2 - diff**2) ** 1.5)) / (
            math.cosh(3.0 * diff / math.sqrt(delta**2 - diff**2)) ** 2
        )
    return 0.0


@njit
def delta_parabolic(u: float, u0: float, delta: float) -> float:
    diff = u - u0
    if abs(diff) <= delta:
        return 0.75 * (1 - diff**2 / delta**2) / delta
    return 0.0


@njit
def delta_box(u: float, u0: float, delta: float) -> float:
    return (0.5 / delta) if abs(u - u0) <= delta else 0.0


def get_step_fn(scheme: StepScheme) -> Callable[[float, float, float], float]:
    return {
        StepScheme.ERF: step_erf,
        StepScheme.HYPER: step_hyper,
        StepScheme.LINEAR: step_linear,
        StepScheme.CONST: step_const,
    }[scheme]


def get_delta_fn(
    scheme: DeltaScheme,
) -> (
    Callable[[float, float, float], float]
    | Callable[[float, float, float, float], float]
):
    return {
        DeltaScheme.GAUSS: delta_gauss,
        DeltaScheme.HYPER: delta_hyper,
        DeltaScheme.PARABOLIC: delta_parabolic,
        DeltaScheme.BOX: delta_box,
    }[scheme]
