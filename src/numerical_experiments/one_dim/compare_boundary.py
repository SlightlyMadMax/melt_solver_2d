import math
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional
from scipy.optimize import fsolve
from scipy.special import erf

from src.constants import K_ICE, K_WATER, C_ICE_VOL, C_WATER_VOL, L_VOL, dy

g = -5.0
u_0 = 5.0

a_ice = (K_ICE / C_ICE_VOL) ** 0.5
a_water = (K_WATER / C_WATER_VOL) ** 0.5


def trans_eq(_gamma: float):
    lhs = K_ICE * g * math.exp(-(_gamma / (2.0 * a_ice)) ** 2) / (a_ice * erf(_gamma / (2.0 * a_ice)))
    rhs = -K_WATER * u_0 * math.exp(-(_gamma / (2.0 * a_water)) ** 2) / \
          (a_water * (1.0 - erf(_gamma / (2.0 * a_water)))) - \
          _gamma * L_VOL * math.pi ** 0.5 / 2
    return lhs - rhs


def compare_num_with_analytic(num: list[float], _s_0: float, dir_name: str,
                              _delta: Optional[float] = None, show_graphs: bool = True):

    gamma = fsolve(trans_eq, 0.0002)[0]

    print(f"GAMMA: {gamma}")

    n = len(num)
    print(f"Modeling time: {n} days.")

    t_0 = (_s_0 / gamma) ** 2

    print(int(t_0/3600))

    print(num)

    time = [i * 60. * 60. * 24.0 + t_0 for i in range(n)]

    exact = [gamma * time[i] ** 0.5 for i in range(n)]

    print(exact)

    relative_error = [abs(exact[i] - num[i]) * 100 / exact[i] for i in range(n)]

    print(relative_error)

    abs_error = [abs(exact[i] - num[i]) for i in range(n)]

    print(abs_error)

    print(f"Average abs. error: {np.average(abs_error)}")
    print(f"Шаг сетки: {dy}")

    fig = plt.figure()

    ax = plt.axes()
    plt.plot(
        time,
        relative_error,
        linewidth=1,
        color='r',
        label="дельта = " + str(_delta) if _delta is not None else "адаптивная дельта"
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
        color='r',
        label="дельта = " + str(_delta) if _delta is not None else "адаптивная дельта"
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
    plt.plot(time, exact, linewidth=1, color='r', label='Аналитическое решение')
    plt.plot(time, num, linewidth=1, color='k', label='Численное решение')
    ax.set_title("Сравнение численного и аналитического решения")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Положение границы фазового перехода, м")
    ax.legend()
    plt.savefig(f"{dir_name}/boundary.png")
    if show_graphs:
        plt.show()
