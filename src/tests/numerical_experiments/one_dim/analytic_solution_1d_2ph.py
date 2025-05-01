import math
import numpy as np

from scipy.optimize import fsolve
from scipy.special import erf

from src.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.parameters.thermal import ThermalParameters


def trans_eq(gamma: float, params: ThermalParameters, min_temp: float, max_temp: float):
    a_ice = params.thermal_diffusivity_solid**0.5
    a_water = params.thermal_diffusivity_liquid**0.5

    lhs = (
        params.thermal_conductivity_solid
        * min_temp
        * math.exp(-((gamma / (2.0 * a_ice)) ** 2))
        / (a_ice * erf(gamma / (2.0 * a_ice)))
    )

    rhs = (
        -params.thermal_conductivity_liquid
        * max_temp
        * math.exp(-((gamma / (2.0 * a_water)) ** 2))
        / (a_water * (1.0 - erf(gamma / (2.0 * a_water))))
        - gamma * params.volumetric_latent_heat * math.pi**0.5 / 2
    )

    return lhs - rhs


def get_ice_temp(
    y: float, gamma: float, t: float, min_temp: float, params: ThermalParameters
) -> float:
    a_ice = params.thermal_diffusivity_solid**0.5

    return min_temp * (
        1.0 - erf(y / (2.0 * a_ice * t**0.5)) / erf(gamma / (2.0 * a_ice))
    )


def get_water_temp(
    y: float, gamma: float, t: float, max_temp: float, params: ThermalParameters
) -> float:
    a_water = params.thermal_diffusivity_liquid**0.5
    return (
        max_temp
        * (erf(y / (2.0 * a_water * t**0.5)) - erf(gamma / (2.0 * a_water)))
        / (1.0 - erf(gamma / (2.0 * a_water)))
    )


def get_analytic_solution(
    t: float,
    min_temp: float,
    max_temp: float,
    geometry: DomainGeometry,
    params: ThermalParameters,
) -> np.ndarray:
    """
    Find the analytical solution of a model one-dimensional two-phase problem with given parameters at a given moment
    of time.

    :param t: Time
    :param min_temp: Initial temperature of the solid phase region.
    :param max_temp: Initial temperature of the liquid phase region.
    :param geometry: Object containing the domain geometry information.
    :param params: Object containing parameters of the problem like thermal conductivity etc.
    :return: Temperature distribution for the model problem at the time corresponding to the position of the free boundary s_0.
    """
    result = np.empty((geometry.n_y, geometry.n_x))

    gamma: float = fsolve(  # noqa
        lambda x: trans_eq(
            gamma=x,
            params=params,
            min_temp=min_temp + ABS_ZERO,
            max_temp=max_temp + ABS_ZERO,
        ),
        0.0002,  # noqa
    )[0]

    s = t**0.5 * gamma

    for j in range(geometry.n_y):
        result[j, :] = (
            get_ice_temp(
                y=j * geometry.dy,
                gamma=gamma,
                t=t,
                min_temp=min_temp + ABS_ZERO,
                params=params,
            )
            if j * geometry.dy <= s
            else get_water_temp(
                y=j * geometry.dy,
                gamma=gamma,
                t=t,
                max_temp=max_temp + ABS_ZERO,
                params=params,
            )
        )

    return result
