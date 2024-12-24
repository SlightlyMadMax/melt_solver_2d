import math
import numba


@numba.jit(nopython=True)
def delta_function(u: float, u_0: float, delta: float) -> float:
    """
    Smoothed approximation of the delta function, centered at u_0.

    :param u: The temperature value (deviation from the reference temperature).
    :param u_0: The point where the delta function is centered.
    :param delta: The smoothing parameter.
    :return: The value of the smoothed delta function at the point u.
    """
    return math.exp(-(u - u_0) * (u - u_0) / (2.0 * delta * delta)) / (
        (2.0 * math.pi) ** 0.5 * delta
    )


@numba.jit(nopython=True)
def c_smoothed(
    u: float,
    u_pt_ref: float,
    c_solid: float,
    c_liquid: float,
    l_solid: float,
    delta: float,
) -> float:
    """
    Smoothed effective volumetric heat capacity.

    :param u: The temperature value (deviation from the reference temperature).
    :param u_pt_ref: The phase transition temperature (deviation from the reference temperature).
    :param c_solid: The volumetric heat capacity of the solid phase.
    :param c_liquid: The volumetric heat capacity of the liquid phase.
    :param l_solid: The volumetric latent heat of fusion of the solid phase.
    :param delta: The smoothing parameter.
    :return: The value of the smoothed effective volumetric heat capacity at the temperature u.
    """
    if delta <= 0:
        return c_solid if u < u_pt_ref else c_liquid

    return (
        c_solid
        + (c_liquid - c_solid)
        * (1.0 + math.erf((u - u_pt_ref) / (2**0.5 * delta)))
        * 0.5
        + l_solid * delta_function(u=u, u_0=u_pt_ref, delta=delta)
    )


@numba.jit(nopython=True)
def k_smoothed(
    u: float,
    u_pt_ref: float,
    k_solid: float,
    k_liquid: float,
    delta: float,
) -> float:
    """
    Smoothed heat conductivity coefficient.

    :param u: The temperature value (deviation from the reference temperature).
    :param u_pt_ref: The phase transition temperature (deviation from the reference temperature).
    :param k_solid: The heat conductivity of the solid phase.
    :param k_liquid: The heat conductivity of the liquid phase.
    :param delta: The smoothing parameter.
    :return: The value of the smoothed heat conductivity coefficient at the temperature u.
    """
    if delta <= 0.0:
        return k_solid if u < u_pt_ref else k_liquid

    return (
        k_solid
        + (k_liquid - k_solid)
        * (1.0 + math.erf((u - u_pt_ref) / (2**0.5 * delta)))
        * 0.5
    )


@numba.jit(nopython=True)
def delta_parabolic(u: float, u_0: float, delta: float) -> float:
    if abs(u - u_0) <= delta:
        return 0.75 * (1.0 - u * u / (delta * delta)) / delta
    return 0.0


@numba.jit(nopython=True)
def delta_const(u: float, u_pt: float, delta: float) -> float:
    if abs(u - u_pt) <= delta:
        return 0.5 / delta
    return 0.0


@numba.jit(nopython=True)
def c_simple(
    u: float,
    u_pt_ref: float,
    c_solid: float,
    c_liquid: float,
    l_solid: float,
    delta: float,
) -> float:
    if abs(u - u_pt_ref) <= delta:
        return 0.5 * (c_liquid + c_solid) + l_solid * delta_const(u, delta)
    elif u < u_pt_ref - delta:
        return c_solid
    else:
        return c_liquid


@numba.jit(nopython=True)
def k_simple(
    u: float,
    u_pt_ref: float,
    k_solid: float,
    k_liquid: float,
    delta: float,
) -> float:
    if abs(u - u_pt_ref) <= delta:
        return k_liquid + 0.5 * (k_liquid - k_solid) * (u - u_pt_ref - delta) / delta
    elif u < u_pt_ref - delta:
        return k_solid
    else:
        return k_liquid
