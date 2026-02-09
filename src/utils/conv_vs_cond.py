import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
        (0.1, 0.92),
        0.035,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(circle)

    ax.text(
        0.1,
        0.92,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
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
ax0.legend()

add_subfigure_label(ax0, "а")


# -------- (б) freezing ----------
ax1.plot(t, b_conv_freeze[0:1441:10], linewidth=2, label="С учётом конвекции")
ax1.plot(t, b_stef_freeze[0:1441:10], "--", linewidth=2, label="Без учёта конвекции")

ax1.set_xlabel("Время, ч")
ax1.set_ylabel("Среднее положение границы, м")
ax1.legend()

add_subfigure_label(ax1, "б")


# -----------------------------
plt.savefig("../../graphs/wavy_surface/boundary_vs_time.jpg", dpi=300)
plt.show()
