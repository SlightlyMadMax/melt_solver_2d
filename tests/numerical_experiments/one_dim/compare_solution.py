import matplotlib.pyplot as plt
import numpy as np

from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.parameters.config import ExperimentConfig
from tests.numerical_experiments.one_dim.analytic_solution_1d_2ph import (
    get_analytic_solution,
)


def calculate_and_plot_interface_error(
    cfg: ExperimentConfig,
    gamma: float,
    num: list[float],
    time: list[float],
    dir_name: str,
    show_graphs: bool = True,
) -> None:
    """
    Calculate and plot the absolute and relative errors of the phase transition boundary position.

    The analytical interface position is assumed to scale as gamma * sqrt(time).

    :param cfg: Experiment configuration containing physical parameters (thermal conductivity, etc.).
    :param gamma: Proportionality coefficient between interface position and sqrt(time).
    :param num: Array of computed interface positions at each time step.
    :param time: Array of model times corresponding to `num`.
    :param dir_name: Directory where the graphs will be saved.
    :param show_graphs: If True, display the graphs in a new window.
    :return: None
    """
    n = len(num)

    exact = [gamma * time[i] ** 0.5 for i in range(n)]

    relative_error = [abs(exact[i] - num[i]) * 100 / exact[i] for i in range(1, n)]

    abs_error = [abs(exact[i] - num[i]) for i in range(n)]

    print(f"Average abs. error of the boundary position: {np.average(abs_error)}\n")

    ax = plt.axes()
    plt.plot(
        time[1:],
        relative_error,
        linewidth=1,
        color="r",
        label=(
            "дельта = " + str(round(cfg.delta_nd * cfg.delta_u, 3))
            if cfg.delta_nd is not None
            else "адаптивная дельта"
        ),
    )
    plt.grid()
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
            "дельта = " + str(round(cfg.delta_nd * cfg.delta_u, 3))
            if cfg.delta_nd is not None
            else "адаптивная дельта"
        ),
    )
    plt.grid()
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
    plt.grid()
    ax.set_title("Сравнение численного и аналитического решения")
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Положение границы фазового перехода, м")
    ax.legend()
    plt.savefig(f"{dir_name}/boundary.png")
    if show_graphs:
        plt.show()


def calculate_and_plot_temperature_error(
    cfg: ExperimentConfig,
    gamma: float,
    num: np.ndarray,
    min_temp: float,
    max_temp: float,
    dir_name: str,
    show_graphs: bool = True,
) -> None:
    """
    Calculate L2 error of the temperature field.

    :param cfg: Experiment configuration containing physical parameters (thermal conductivity, etc.).
    :param gamma: Proportionality coefficient between interface position and sqrt(time).
    :param num: Array of computed temperature fields at each time step.
    :param min_temp: Initial temperature of the solid phase region.
    :param max_temp: Initial temperature of the liquid phase region.
    :param dir_name: Directory where the graphs will be saved.
    :param show_graphs: If True, display the graphs in a new window.
    :return: None
    """
    geometry: DomainGeometry = cfg.geometry
    dim_analytical = get_analytic_solution(
        cfg=cfg,
        t=geometry.n_t * geometry.dt,
        gamma=gamma,
        min_temp=min_temp,
        max_temp=max_temp,
    )
    non_dim_analytical = (dim_analytical - ABS_ZERO - cfg.u_ref) / cfg.delta_u
    dim_num = num * cfg.delta_u + cfg.u_ref + ABS_ZERO
    y = np.linspace(0, geometry.height, geometry.n_y)

    center_index = int(geometry.n_x / 2)
    temp_top = non_dim_analytical[-1, center_index] * cfg.delta_u + ABS_ZERO + cfg.u_ref
    temp_near_top = (
        non_dim_analytical[-2, center_index] * cfg.delta_u + ABS_ZERO + cfg.u_ref
    )
    print(f"Temperature at and near the top boundary: {temp_top} C, {temp_near_top} C")

    L2_error = np.linalg.norm(
        num[1:-1, 1:-1] - non_dim_analytical[1:-1, 1:-1]
    ) / np.sqrt(num[1:-1, 1:-1].size)
    print(f"L2 temperature error: {L2_error}")

    ax = plt.axes()
    plt.plot(
        y,
        dim_analytical[:, center_index],
        linewidth=1,
        label="Analytical",
    )
    plt.plot(
        y,
        dim_num[:, center_index],
        linewidth=1,
        label="Numerical",
    )
    plt.grid()
    ax.set_title("Temperature distribution")
    ax.set_xlabel("Y, m")
    ax.set_ylabel("Temperature, C")
    ax.legend()
    plt.savefig(f"{dir_name}/temperature_profile.png")
    if show_graphs:
        plt.show()
