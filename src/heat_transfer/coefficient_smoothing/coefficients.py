import math
from enum import Enum
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
def step_erf(u, u0, delta):
    return 0.5 * (1.0 + math.erf((u - u0) / (math.sqrt(2) * delta)))


@njit
def step_hyper(u, u0, delta):
    diff = u - u0
    if abs(diff) < delta:
        return 0.5 * (1.0 + math.tanh(3.0 * diff / math.sqrt(delta**2 - diff**2)))
    return 1.0 if u > u0 else 0.0


@njit
def step_lin(u, u0, delta):
    diff = u - u0
    if diff >= delta:
        return 1.0
    elif diff <= -delta:
        return 0.0
    return (diff + delta) / (2.0 * delta)


@njit
def step_const(u, u0, delta):
    diff = u - u0
    if diff >= delta:
        return 1.0
    elif diff <= -delta:
        return 0.0
    return 0.5


@njit
def delta_gauss(u, u0, delta):
    return math.exp(-((u - u0) ** 2) / (2 * delta**2)) / (
        math.sqrt(2 * math.pi) * delta
    )


@njit
def delta_hyper(u, u0, delta):
    diff = u - u0
    if abs(diff) < delta:
        return (1.5 * (delta**2 / (delta**2 - diff**2) ** 1.5)) / (
            math.cosh(3.0 * diff / math.sqrt(delta**2 - diff**2)) ** 2
        )
    return 0.0


@njit
def delta_parabolic(u, u0, delta):
    diff = u - u0
    if abs(diff) <= delta:
        return 0.75 * (1 - diff**2 / delta**2) / delta
    return 0.0


@njit
def delta_box(u, u0, delta):
    return (0.5 / delta) if abs(u - u0) <= delta else 0.0


def get_step_fn(scheme: StepScheme):
    return {
        StepScheme.ERF: step_erf,
        StepScheme.HYPER: step_hyper,
        StepScheme.LINEAR: step_lin,
        StepScheme.CONST: step_const,
    }[scheme]


def get_delta_fn(scheme: DeltaScheme):
    return {
        DeltaScheme.GAUSS: delta_gauss,
        DeltaScheme.HYPER: delta_hyper,
        DeltaScheme.PARABOLIC: delta_parabolic,
        DeltaScheme.BOX: delta_box,
    }[scheme]
