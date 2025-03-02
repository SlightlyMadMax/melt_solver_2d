import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import fsolve
from scipy.special import erf

from src.constants import ABS_ZERO
from src.heat_transfer.parameters import ThermalParameters


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
            - gamma * params.volumetric_latent_heat * math.pi ** 0.5 / 2
    )
    return lhs - rhs


def compare_num_with_analytic(
    min_temp: float,
    max_temp: float,
    params: ThermalParameters,
    num: list[float],
    s_0: float,
    dir_name: str,
    show_graphs: bool = True,
) -> None:
    """

    :param min_temp: Initial temperature of the solid phase region.
    :param max_temp: Initial temperature of the liquid phase region.
    :param params: Object containing parameters of the problem like thermal conductivity etc.
    :param num: Array containing positions of the boundary throughout the modelling time.
    :param s_0: Initial position of the boundary.
    :param dir_name: Name of the directory where the graphs will be saved.
    :param show_graphs: If set to True, the graphs will be opened in a new window.
    :return: None
    """

    gamma = fsolve(
        lambda x: trans_eq(
            gamma=x,
            params=params,
            min_temp=min_temp + ABS_ZERO,
            max_temp=max_temp + ABS_ZERO,
        ),
        0.0002,
    )[0]

    n = len(num)

    t_0: float = (s_0 / gamma) ** 2

    print(int(t_0 / 3600))

    time = [i * 60.0 * 60.0 * 24.0 + t_0 for i in range(n)]

    exact = [gamma * time[i] ** 0.5 for i in range(n)]

    relative_error = [abs(exact[i] - num[i]) * 100 / exact[i] for i in range(n)]

    abs_error = [abs(exact[i] - num[i]) for i in range(n)]

    print(f"Average abs. error: {np.average(abs_error)}\n")

    fig = plt.figure()

    ax = plt.axes()
    plt.plot(
        time,
        relative_error,
        linewidth=1,
        color="r",
        label=(
            "дельта = " + str(params.delta)
            if params.delta is not None
            else "адаптивная дельта"
        ),
    )
    ax.set_title("Относительная погрешность")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Относительная погрешность, %")
    ax.legend()
    plt.savefig(f"{dir_name}/boundary_rel_error.png")
    if show_graphs:
        plt.show()
    plt.clf()

    ax = plt.axes()
    plt.plot(
        time,
        abs_error,
        linewidth=1,
        color="r",
        label=(
            "дельта = " + str(params.delta)
            if params.delta is not None
            else "адаптивная дельта"
        ),
    )
    ax.set_title("Абсолютная погрешность")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Абсолютная погрешность, м")
    ax.legend()
    plt.savefig(f"{dir_name}/boundary_abs_error.png")
    if show_graphs:
        plt.show()
    plt.clf()

    ax = plt.axes()
    plt.plot(time, exact, linewidth=1, color="r", label="Аналитическое решение")
    plt.plot(time, num, linewidth=1, color="k", label="Численное решение")
    ax.set_title("Сравнение численного и аналитического решения")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Положение границы фазового перехода, м")
    ax.legend()
    plt.savefig(f"{dir_name}/boundary.png")
    if show_graphs:
        plt.show()
