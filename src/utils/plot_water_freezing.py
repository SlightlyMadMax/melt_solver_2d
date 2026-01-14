import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.init_values import initialize_velocity
from src.fluid_dynamics.utils import calculate_velocity_from_sf
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


# -----------------------------
# helper for subfigure labels
# -----------------------------
def add_subfigure_label(ax, label):
    circle = patches.Circle(
        (0.06, 0.94),
        0.045,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(circle)

    ax.text(
        0.06,
        0.94,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        zorder=11,
    )


# -----------------------------
# load data
# -----------------------------
cfg: ExperimentConfig = ExperimentConfig.load_from_file("../../parameter_sets/water/freezing.json")
geometry: DomainGeometry = cfg.geometry
img = plt.imread("../../data/kowalewski.png")
data = np.load("../../data/water_freezing/checkpoint_234000_v3.npz")
u = data["u"]
sf = data["sf"]
w = data["w"]
u_dim = u * cfg.delta_u + cfg.u_ref
v_x, v_y = initialize_velocity(geometry=geometry)
calculate_velocity_from_sf(sf, v_x, v_y, cfg)

n_x, n_y = u.shape[1], u.shape[0]
x = np.linspace(0, geometry.width, n_x)
y = np.linspace(0, geometry.height, n_y)
X, Y = np.meshgrid(x, y)

# -----------------------------
# figure
# -----------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

# -------- (а) field ----------
ax0.imshow(img, extent=[0, geometry.width, 0, geometry.height])
X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u_dim)
ax0.plot(X_b, Y_b, linestyle="--", color="red", linewidth=2, label="Численное решение")
ax0.legend()

ax0.set_xlabel("x, м")
ax0.set_ylabel("y, м")
ax0.set_aspect("equal", adjustable="box")

add_subfigure_label(ax0, "а")

# -------- (б) profile --------
stride = 8
contour = ax1.contourf(X, Y, u_dim + ABS_ZERO, levels=101, cmap="Blues")

ax1.quiver(
    X[::stride, ::stride],
    Y[::stride, ::stride],
    v_x[::stride, ::stride],
    v_y[::stride, ::stride],
    color="black",
    scale_units="xy",
)

ax1.plot(X_b, Y_b, linestyle="--", color="k", linewidth=1.5)

ax1.set_xlabel("x, м")
ax1.set_ylabel("y, м")
ax1.set_aspect("equal", adjustable="box")

cbar = fig.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_ticks(np.linspace(-10, 10, 9))
cbar.set_label(r"Температура, $^{\circ}\mathrm{C}$", rotation=270, labelpad=15)

add_subfigure_label(ax1, "б")

# -----------------------------
plt.savefig("../../graphs/velocity/bruh.jpg", dpi=300)
plt.show()
