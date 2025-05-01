import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from numpy.typing import NDArray

from src.core.geometry import DomainGeometry


def plot_velocity_field(
    v_x: NDArray[np.float64],
    v_y: NDArray[np.float64],
    geometry: DomainGeometry,
    graph_id: int,
    show_graph: bool = True,
    directory: str = "../graphs/velocity/",
    equal_aspect: Optional[bool] = True,
):
    n_y, n_x = v_x.shape
    dx, dy = geometry.dx, geometry.dy
    x = np.linspace(0, (n_x - 1) * dx, n_x)
    y = np.linspace(0, (n_y - 1) * dy, n_y)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, v_x, v_y, angles="xy", scale_units="xy", scale=1, color="blue")

    plt.xlabel("x, м")
    plt.ylabel("y, м")
    plt.title("Поле скоростей")

    if equal_aspect:
        plt.axis("equal")

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}v_{graph_id}.png")

    if show_graph:
        plt.show()
    else:
        plt.close()


def plot_stream_function(
    stream_function: NDArray[np.float64],
    geometry: DomainGeometry,
    graph_id: int,
    show_graph: bool = True,
    directory: str = "../graphs/stream_function/",
    equal_aspect: Optional[bool] = True,
):
    n_y, n_x = stream_function.shape
    dx, dy = geometry.dx, geometry.dy
    x = np.linspace(0, (n_x - 1) * dx, n_x)
    y = np.linspace(0, (n_y - 1) * dy, n_y)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, stream_function, levels=15, cmap='viridis')
    plt.clabel(cp, inline=True, fmt="$\psi$ = %.2f", fontsize=10)

    plt.xlabel("x, м")
    plt.ylabel("y, м")
    # plt.title("Линии тока")

    if equal_aspect:
        plt.axis("equal")

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{directory}stream_function_{graph_id}.png")

    if show_graph:
        plt.show()
    else:
        plt.close()
