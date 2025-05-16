import math
from enum import Enum
from numba import njit


class StepScheme(Enum):
    ERF = "erf"
    HYPER = "hyper"


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
    if scheme is StepScheme.ERF:
        return step_erf
    elif scheme is StepScheme.HYPER:
        return step_hyper
    else:
        raise ValueError("Unknown step scheme")


def get_delta_fn(scheme: DeltaScheme):
    return {
        DeltaScheme.GAUSS: delta_gauss,
        DeltaScheme.HYPER: delta_hyper,
        DeltaScheme.PARABOLIC: delta_parabolic,
        DeltaScheme.BOX: delta_box,
    }[scheme]


@njit
def c_smoothed(
    u: float,
    u_pt: float,
    c_solid: float,
    c_liquid: float,
    l_solid: float,
    delta: float,
    step_fn: callable,
    delta_fn: callable,
) -> float:
    if delta <= 0:
        return c_solid if u < u_pt else c_liquid
    return (
        c_solid
        + (c_liquid - c_solid) * step_fn(u, u_pt, delta)
        + l_solid * delta_fn(u, u_pt, delta)
    )


@njit
def k_smoothed(
    u: float,
    u_pt: float,
    k_solid: float,
    k_liquid: float,
    delta: float,
    step_fn: callable,
) -> float:
    if delta <= 0:
        return k_solid if u < u_pt else k_liquid
    return k_solid + (k_liquid - k_solid) * step_fn(u, u_pt, delta)
