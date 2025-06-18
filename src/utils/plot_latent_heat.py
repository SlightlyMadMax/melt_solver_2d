import numpy as np
from matplotlib import pyplot as plt

from src.heat_transfer.coefficient_smoothing.coefficients import delta_gauss
from src.heat_transfer.coefficient_smoothing.mushy_zone import get_dilated_mushy_mask
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig


def latent_heat_density_field(
    u: np.ndarray, u_pt: float, delta: np.ndarray, l_solid: float
) -> np.ndarray:
    n_y, n_x = u.shape
    val = np.zeros_like(u)
    mushy_mask = get_dilated_mushy_mask(u_dim=u, u_pt=u_pt, delta=delta, extend_by=1)
    for j in range(n_y):
        for i in range(n_x):
            if mushy_mask[j, i]:
                val[j, i] = delta_gauss(u[j, i], u_pt, delta[j, i])
    return l_solid * val


def plot_latent_heat_field(
    u: np.ndarray,
    cfg: ExperimentConfig,
    delta: np.ndarray,
    l_solid: float,
    graph_id: int,
    directory: str = "../graphs/latent_heat/",
):
    latent_heat_field = latent_heat_density_field(
        u, u_pt=cfg.material_props.u_pt, delta=delta, l_solid=l_solid
    )
    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u)
    X, Y = cfg.geometry.mesh_grid

    plt.figure(figsize=(8, 6))

    ax = plt.axes(
        xlim=(0, cfg.geometry.width),
        ylim=(0, cfg.geometry.height),
        xlabel="x, м",
        ylabel="y, м",
    )
    ax.set_aspect("equal")

    contour = plt.contourf(
        X,
        Y,
        latent_heat_field,
        25,
        cmap="viridis",
    )
    cbar = plt.colorbar(
        contour, label="Величина слагаемого, отвечающего за скрытую теплоту плавления"
    )

    for i in range(X.shape[0]):
        plt.plot(X[i, :], Y[i, :], color="white", linewidth=0.3, alpha=0.6)
    for j in range(X.shape[1]):
        plt.plot(X[:, j], Y[:, j], color="white", linewidth=0.3, alpha=0.6)

    plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color="r", label="Граница ф.п.")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{directory}l_{graph_id}.png")

    plt.close()
