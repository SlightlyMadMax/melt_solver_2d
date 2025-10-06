import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from numpy.typing import NDArray

from src.core.geometry import DomainGeometry
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig


def plot_velocity_field(
        v_x: NDArray[np.float64],
        v_y: NDArray[np.float64],
        u_dim: NDArray[np.float64],
        cfg: ExperimentConfig,
        graph_id: int,
        show_graph: bool = True,
        plot_boundary: bool = True,
        directory: str = "../graphs/velocity/",
        equal_aspect: Optional[bool] = True,
        stride: int = 8,
):
    geometry: DomainGeometry = cfg.geometry
    X, Y = geometry.mesh_grid

    X_sub = X[::stride, ::stride]
    Y_sub = Y[::stride, ::stride]
    v_x_sub = v_x[::stride, ::stride]
    v_y_sub = v_y[::stride, ::stride]

    plt.figure(figsize=(8, 6))

    contour = plt.contourf(
        X,
        Y,
        u_dim,
        25,
        cmap="viridis",
        extend="both",
    )
    cbar = plt.colorbar(contour)
    cbar.set_ticks(np.linspace(np.min(u_dim), np.max(u_dim), num=6))
    cbar.set_label("Температура", rotation=270, labelpad=15, fontsize=14)

    plt.quiver(
        X_sub,
        Y_sub,
        v_x_sub,
        v_y_sub,
        angles="xy",
        scale_units="xy",
        scale=0.2,
        color="black",
        width=0.003,
    )
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    # plt.title("Поле скоростей")

    if plot_boundary:
        X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u_dim)
        plt.plot(X_b, Y_b, linestyle="--", color="red", linewidth=2)

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
    X, Y = geometry.mesh_grid

    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, stream_function, levels=15, cmap="viridis")
    plt.clabel(cp, inline=True, fmt=r"$\psi$ = %.2E", fontsize=10)

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
