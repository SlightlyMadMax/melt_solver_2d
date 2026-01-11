import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.parameters.config import ExperimentConfig
from src.utils.water_convection_benchmark import calculate_T_profile_Y05

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
cfg = ExperimentConfig.load_from_file("../../parameter_sets/water/convection.json")

data = np.load("../../data/water_convection/conv.npz")
u = data["u"]
v_x, v_y = data["v_x"], data["v_y"]

n_x, n_y = u.shape[1], u.shape[0]
x = np.linspace(0, 1, n_x)
y = np.linspace(0, 1, n_y)
X, Y = np.meshgrid(x, y)

# -----------------------------
# figure
# -----------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

# -------- (а) field ----------
stride = 8
levels = np.linspace(0.0, 1.0, 101)
contour = ax0.contourf(X, Y, u, levels=levels, cmap="Blues")

ax0.quiver(
    X[::stride, ::stride],
    Y[::stride, ::stride],
    v_x[::stride, ::stride],
    v_y[::stride, ::stride],
    color="black",
    scale_units="xy",
)

ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.set_xlabel("X")
ax0.set_ylabel("Y")
ax0.set_aspect("equal", adjustable="box")

cbar = fig.colorbar(contour, ax=ax0, fraction=0.046, pad=0.04)
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.set_label(r"$\theta$", rotation=270, labelpad=15)

add_subfigure_label(ax0, "а")

# -------- (б) profile --------
u_true = calculate_T_profile_Y05(x)
u_mid = u[n_y // 2, :]

ax1.plot(x, u_true, label="Michalek et al., 2005", linewidth=2)
ax1.plot(x, u_mid, "--", label="Численное решение", linewidth=2)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel("X")
ax1.set_ylabel(r"$\theta$")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.set_aspect("equal", adjustable="box")
ax1.legend()

add_subfigure_label(ax1, "б")

# -----------------------------
plt.savefig("../../graphs/velocity/validation_combined.jpg", dpi=300)
plt.show()
