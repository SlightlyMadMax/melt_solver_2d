import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.parameters.config import ExperimentConfig
from src.examples.water_convection.benchmark_solution import calculate_T_profile_Y05

mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
)


# -----------------------------
# helper for subfigure labels
# -----------------------------
def add_subfigure_label(ax, label):
    circle = patches.Circle(
        (0.06, 0.94),
        0.035,
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
        fontsize=14,
        zorder=11,
    )


# -----------------------------
# load data
# -----------------------------
cfg = ExperimentConfig.load_from_file("./config.json")

data = np.load("./data/1st_order_bc/151x151/checkpoint_7200.npz")
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
ax0.set_xlabel(r"$X$")
ax0.set_ylabel(r"$Y$")
ax0.set_aspect("equal", adjustable="box")

cbar = fig.colorbar(contour, ax=ax0, fraction=0.046, pad=0.04)
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.set_label("Безразмерная температура", rotation=270, labelpad=15)

add_subfigure_label(ax0, "а")

# -------- (б) profile --------
u_true = calculate_T_profile_Y05(x)
u_mid = u[n_y // 2, :]

ax1.plot(x, u_true, label="Michalek et al., 2005", linewidth=2)
ax1.plot(x, u_mid, "--", label="Численное решение", linewidth=2)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel(r"$X$")
ax1.set_ylabel("Θ")
ax1.set_aspect("equal", adjustable="box")
ax1.legend()

add_subfigure_label(ax1, "б")

# -----------------------------
plt.savefig("./graphs/compared.tif", dpi=300)
plt.show()
