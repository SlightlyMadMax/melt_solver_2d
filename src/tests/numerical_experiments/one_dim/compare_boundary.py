import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import fsolve

from src.core.constants import ABS_ZERO
from src.parameters.thermal import ThermalParameters
from src.tests.numerical_experiments.one_dim.analytic_solution_1d_2ph import trans_eq


def compare_num_with_analytic(
    min_temp: float,
    max_temp: float,
    params: ThermalParameters,
    num: list[float],
    time: list[float],
    dir_name: str,
    show_graphs: bool = True,
) -> None:
    """

    :param min_temp: Initial temperature of the solid phase region.
    :param max_temp: Initial temperature of the liquid phase region.
    :param params: Object containing parameters of the problem like thermal conductivity etc.
    :param num: Array containing positions of the boundary throughout the modelling time.
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

    exact = [gamma * time[i] ** 0.5 for i in range(n)]

    relative_error = [abs(exact[i] - num[i]) * 100 / exact[i] for i in range(1, n)]

    abs_error = [abs(exact[i] - num[i]) for i in range(n)]

    print(f"Average abs. error: {np.average(abs_error)}\n")

    fig = plt.figure()

    ax = plt.axes()
    plt.plot(
        time[1:],
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
