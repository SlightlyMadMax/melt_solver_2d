import math
import numpy as np

from scipy.optimize import fsolve
from scipy.special import erf

from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties


def trans_eq(
    gamma: float, material_props: MaterialProperties, min_temp: float, max_temp: float
):
    a_ice = material_props.thermal_diffusivity_solid**0.5
    a_water = material_props.thermal_diffusivity_liquid**0.5

    lhs = (
        material_props.thermal_conductivity_solid
        * min_temp
        * math.exp(-((gamma / (2.0 * a_ice)) ** 2))
        / (a_ice * erf(gamma / (2.0 * a_ice)))
    )

    rhs = (
        -material_props.thermal_conductivity_liquid
        * max_temp
        * math.exp(-((gamma / (2.0 * a_water)) ** 2))
        / (a_water * (1.0 - erf(gamma / (2.0 * a_water))))
        - gamma * material_props.volumetric_latent_heat * math.pi**0.5 / 2
    )

    return lhs - rhs


def get_ice_temp(
    y: float,
    gamma: float,
    t: float,
    min_temp: float,
    material_props: MaterialProperties,
) -> float:
    a_ice = material_props.thermal_diffusivity_solid**0.5

    return min_temp * (
        1.0 - erf(y / (2.0 * a_ice * t**0.5)) / erf(gamma / (2.0 * a_ice))
    )


def get_water_temp(
    y: float,
    gamma: float,
    t: float,
    max_temp: float,
    material_props: MaterialProperties,
) -> float:
    a_water = material_props.thermal_diffusivity_liquid**0.5
    return (
        max_temp
        * (erf(y / (2.0 * a_water * t**0.5)) - erf(gamma / (2.0 * a_water)))
        / (1.0 - erf(gamma / (2.0 * a_water)))
    )


def get_analytic_solution(
    cfg: ExperimentConfig,
    t: float,
    gamma: float,
    min_temp: float,
    max_temp: float,
) -> np.ndarray:
    """
    Find the analytical solution of a model one-dimensional two-phase problem with given parameters at a given moment
    of time.

    :param cfg: Experiment configuration (domain geometry, material properties, etc.).
    :param t: Time
    :param gamma: Proportionality coefficient between interface position and sqrt(time).
    :param min_temp: Initial temperature of the solid phase region.
    :param max_temp: Initial temperature of the liquid phase region.
    :return: Temperature distribution for the model problem at the time corresponding to the position of the free boundary s_0.
    """
    geometry: DomainGeometry = cfg.geometry
    material_props: MaterialProperties = cfg.material_props
    result = np.empty((geometry.n_y, geometry.n_x))

    s = t**0.5 * gamma

    for j in range(geometry.n_y):
        result[j, :] = (
            get_ice_temp(
                y=j * geometry.dy,
                gamma=gamma,
                t=t,
                min_temp=min_temp + ABS_ZERO,
                material_props=material_props,
            )
            if j * geometry.dy <= s
            else get_water_temp(
                y=j * geometry.dy,
                gamma=gamma,
                t=t,
                max_temp=max_temp + ABS_ZERO,
                material_props=material_props,
            )
        )

    return result


def calculate_gamma(cfg: ExperimentConfig, min_temp: float, max_temp: float) -> float:
    material_props: MaterialProperties = cfg.material_props
    gamma: float = fsolve(  # noqa
        lambda x: trans_eq(
            gamma=x,
            material_props=material_props,
            min_temp=min_temp + ABS_ZERO,
            max_temp=max_temp + ABS_ZERO,
        ),
        0.0002,  # noqa
    )[0]

    residual = trans_eq(gamma, material_props, min_temp + ABS_ZERO, max_temp + ABS_ZERO)
    print(f"Gamma: {gamma}, Residual: {residual}")

    return gamma
