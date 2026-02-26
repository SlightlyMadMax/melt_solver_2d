import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
        (0.12, 0.92),
        0.03,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(circle)

    ax.text(
        0.12,
        0.92,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        zorder=11,
    )


# -----------------------------
# load data: MELTING
# -----------------------------
b_conv_melt = np.load(
    "../../data/wavy_surface/boundary/melting/convection_boundary.npz"
)["b"]

b_stef_melt = np.load(
    "../../data/wavy_surface/boundary/melting/stefan_boundary.npz"
)["b"]


# -----------------------------
# load data: FREEZING
# -----------------------------
b_conv_freeze = np.load(
    "../../data/wavy_surface/boundary/freezing/convection_boundary.npz"
)["b"]

b_stef_freeze = np.load(
    "../../data/wavy_surface/boundary/freezing/stefan_boundary.npz"
)["b"]


# -----------------------------
# time axis (index → time step)
# -----------------------------
t = np.arange(0, len(b_stef_melt), 10) / 60


# -----------------------------
# figure
# -----------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# -------- (а) melting ----------
ax0.plot(t, b_conv_melt[0:1441:10], linewidth=2, label="С учётом конвекции")
ax0.plot(t, b_stef_melt[0:1441:10], "--", linewidth=2, label="Без учёта конвекции")

ax0.set_xlabel("Время, ч")
ax0.set_ylabel("Среднее положение границы, м")
ax0.set_ylim(0, 0.05)
ax0.legend()

add_subfigure_label(ax0, "а")


# -------- (б) freezing ----------
ax1.plot(t, b_conv_freeze[0:1441:10], linewidth=2, label="С учётом конвекции")
ax1.plot(t, b_stef_freeze[0:1441:10], "--", linewidth=2, label="Без учёта конвекции")

ax1.set_xlabel("Время, ч")
ax1.set_ylabel("Среднее положение границы, м")
ax1.set_ylim(0, 0.05)
ax1.legend()

add_subfigure_label(ax1, "б")


# -----------------------------
plt.savefig("../../graphs/wavy_surface/boundary_vs_time.tif", dpi=300)
plt.show()
