import numpy as np
from enum import Enum

from numba import njit
from math import sin, cos, pi

import src.core.constants as cfg


class TemperatureUnit(Enum):
    CELSIUS = 1, "Celsius"
    KELVIN = 2, "Kelvin"


###
# Robin boundary condition with solar radiation
###


@njit
def get_psi(time: float, n: int) -> np.ndarray:
    # Определяем температуру воздуха у поверхности
    T_air_t = air_temperature(time)

    # Определяем тепловой поток солнечной энергии
    Q_sol = solar_heat(time)

    # Давление насыщенного водяного пара
    p = cfg.A * T_air_t + cfg.B

    # Солнечная радиация с учетом облачности
    h_c = Q_sol * (1.0 - 0.38 * cfg.CLOUDINESS * (1.0 + cfg.CLOUDINESS))

    # Коэффициент теплообмена с воздухом
    c = conv_coef(cfg.WIND_SPEED)

    # Приведенный коэффициент теплообмена
    c_r = c * (1.0 + 0.0195 * cfg.A) + 0.205 * (T_air_t / 100.0) ** 3

    # Приведенная температура окружающей среды
    T_r = (
        c * (T_air_t - 0.0195 * (cfg.B - p * cfg.REL_HUMIDITY))
        + 19.9 * (T_air_t / 100.0) ** 4
        + h_c
    ) / c_r

    psi = T_r * c_r / cfg.K_ICE

    return psi * np.ones(n)


@njit
def get_phi(time: float, n: int) -> np.ndarray:
    # Определяем температуру воздуха у поверхности
    T_air_t = air_temperature(time)

    # Коэффициент теплообмена с воздухом
    c = conv_coef(cfg.WIND_SPEED)

    # Приведенный коэффициент теплообмена
    c_r = c * (1.0 + 0.0195 * cfg.A) + 0.205 * (T_air_t / 100.0) ** 3

    phi = -c_r / cfg.K_ICE

    return phi * np.ones(n)


@njit
def solar_heat(t: float):
    """
    Функция для вычисления потока солнечной радиации.

    :param t: Время в секундах.
    :return: Величина солнечного потока радиации на горизонтальную поверхность при заданных параметрах
    в момент времени t. [Вт/м^2]
    """
    # вычисляем склонение солнца
    decl = cfg.DECL * sin(2 * pi * t / (365 * 24.0 * 3600.0) - pi / 2)

    return cfg.Q_SOL * (
        sin(cfg.LAT) * sin(decl)
        + cos(cfg.LAT) * cos(decl) * cos(cfg.RAD_SPEED * t + 12.0 * 3600.0)
    )


@njit
def air_temperature(t: float):
    """
    Функция изменения температуры воздуха.

    :param t: Время в секундах
    :return: Температура воздуха в заданный момент времени
    """
    return (
        cfg.T_air
        + cfg.T_amp_day * sin(2 * pi * t / (24.0 * 3600.0) + pi / 2)
        + cfg.T_amp_year * sin(2 * pi * t / (365 * 24.0 * 3600.0) + pi / 2)
    )


@njit
def conv_coef(wind_speed: float):
    return wind_speed**0.5 * (7.0 + 7.2 / wind_speed**2)
