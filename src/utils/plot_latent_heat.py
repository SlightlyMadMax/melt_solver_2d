import numpy as np
from matplotlib import pyplot as plt

from src.core.geometry import DomainGeometry
from src.heat_transfer.pt_boundary import get_phase_trans_boundary


def latent_heat_density_field(
    u: np.ndarray, u_pt: float, delta: np.ndarray, l_solid: float
) -> np.ndarray:
    val = np.zeros_like(u)
    mask = np.abs(u - u_pt) < delta

    u_masked = u[mask]
    delta_masked = delta[mask]
    diff = u_masked - u_pt
    denom = (delta_masked**2 - diff**2) ** 1.5
    sech_sq = 1 / np.cosh(3.0 * diff / np.sqrt(delta_masked**2 - diff**2)) ** 2

    val[mask] = 1.5 * delta_masked**2 / denom * sech_sq
    return l_solid * val


def plot_latent_heat_field(
    u: np.ndarray,
    u_pt: float,
    delta: np.ndarray,
    l_solid: float,
    geometry: DomainGeometry,
):
    latent_heat_field = latent_heat_density_field(
        u, u_pt=u_pt, delta=delta, l_solid=l_solid
    )
    X_b, Y_b = get_phase_trans_boundary(
        u=u,
        geom=geometry,
        u_pt=u_pt,
    )
    X, Y = geometry.mesh_grid

    plt.figure(figsize=(8, 6))

    ax = plt.axes(
        xlim=(0, geometry.width),
        ylim=(0, geometry.height),
        xlabel="x, м",
        ylabel="y, м",
    )

    contour = plt.contourf(
        X,
        Y,
        latent_heat_field,
        25,
        cmap="viridis",
    )
    cbar = plt.colorbar(contour, label="Latent Heat Source")

    for i in range(X.shape[0]):
        plt.plot(X[i, :], Y[i, :], color='white', linewidth=0.3, alpha=0.6)
    for j in range(X.shape[1]):
        plt.plot(X[:, j], Y[:, j], color='white', linewidth=0.3, alpha=0.6)

    plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color="r", label="Граница ф.п.")
    ax.legend()
    plt.title("Mushy Zone Visualization (Latent Heat Release)")
    plt.tight_layout()
    plt.show()
