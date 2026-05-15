import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter

from src.core.constants import ABS_ZERO
from src.parameters.config import ExperimentConfig


# --------------------------------------------------
# Настройки matplotlib
# --------------------------------------------------

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


# --------------------------------------------------
# Загрузка конфигураций
# --------------------------------------------------

cfg_4c: ExperimentConfig = ExperimentConfig.load_from_file("./parameters/4c.json")
cfg_5pt6c: ExperimentConfig = ExperimentConfig.load_from_file("./parameters/5pt6c.json")
cfg_8c: ExperimentConfig = ExperimentConfig.load_from_file("./parameters/8c.json")


# --------------------------------------------------
# Универсальная функция подготовки поля
# --------------------------------------------------


def prepare_field(u_dim, geometry, x_min, x_max):
    ny, nx = u_dim.shape

    x = np.linspace(0.0, geometry.width, nx)
    y = np.linspace(0.0, geometry.height, ny)

    mask = (x >= x_min) & (x <= x_max)

    u_crop = u_dim[:, mask]
    X, Y = np.meshgrid(x[mask] - x_min, y)

    return X, Y, u_crop


# --------------------------------------------------
# Описание всех экспериментов и checkpoint'ов
# --------------------------------------------------

experiments = {
    "4c": {
        "cfg": cfg_4c,
        "paths": [
            "./data/4c/checkpoint_0.npz",
            "./data/4c/checkpoint_1800.npz",
            "./data/4c/checkpoint_3600.npz",
        ],
        "title": r"$T_\infty = 4^\circ\mathrm{C}$",
    },
    "5pt6c": {
        "cfg": cfg_5pt6c,
        "paths": [
            "./data/5pt6c/5pt6c_thin/checkpoint_0.npz",
            "./data/5pt6c/5pt6c_thin/checkpoint_9000.npz",
            "./data/5pt6c/5pt6c_v2/checkpoint_28800.npz",
        ],
        "title": r"$T_\infty = 5.6^\circ\mathrm{C}$",
    },
    "8c": {
        "cfg": cfg_8c,
        "paths": [
            "./data/8c/checkpoint_0.npz",
            "./data/8c/checkpoint_750.npz",
            "./data/8c/checkpoint_2100.npz",
        ],
        "title": r"$T_\infty = 8^\circ\mathrm{C}$",
    },
}


# --------------------------------------------------
# Загрузка и подготовка всех данных
# --------------------------------------------------

fields = {}

for key, exp in experiments.items():
    cfg = exp["cfg"]
    fields[key] = []

    for path in exp["paths"]:
        data = np.load(path)
        u = data["u"]
        u_dim = u * cfg.delta_u + cfg.u_ref + ABS_ZERO
        if key == "5pt6c":
            X, Y, u_crop = prepare_field(u_dim, cfg.geometry, 0.2, 0.4)
        else:
            X, Y, u_crop = prepare_field(u_dim, cfg.geometry, 0.3, 0.5)
        fields[key].append((X, Y, u_crop))


# --------------------------------------------------
# Общая цветовая шкала
# --------------------------------------------------

all_u = [u for key in fields for (_, _, u) in fields[key]]

u_min = min(u.min() for u in all_u)
u_max = max(u.max() for u in all_u)

levels = np.linspace(u_min, u_max, 60)
ticks = np.linspace(u_min, u_max, 5)


# --------------------------------------------------
# Построение 3x3 графика
# --------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)

keys = ["4c", "5pt6c", "8c"]

letters = [
    ["а", "б", "в"],
    ["г", "д", "е"],
    ["ж", "з", "и"],
]

for col, key in enumerate(keys):
    exp = experiments[key]
    cfg = exp["cfg"]

    for row in range(3):
        ax = axes[row, col]
        X, Y, u = fields[key][row]

        cf = ax.contourf(X, Y, u, levels=levels, cmap="Blues")

        ax.set_aspect("equal")

        ax.set_xlim(0.0, 0.2)
        ax.set_ylim(0.0, cfg.geometry.height)

        ax.set_xticks([0.0, 0.1, 0.2])
        ax.set_yticks(np.linspace(0.0, cfg.geometry.height, 5))

        # Подписи осей
        if row == 2:
            ax.set_xlabel(r"$x$, м")
        else:
            ax.set_xlabel("")

        if col == 0:
            ax.set_ylabel(r"$y$, м")
        else:
            ax.set_ylabel("")

        # Заголовки только в верхней строке
        if row == 0:
            ax.set_title(exp["title"])

        ax.tick_params(labelsize=10)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # Кружок с буквой
        circle = Circle(
            (0.06, 0.92),
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
            0.92,
            letters[row][col],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            zorder=11,
        )

        # Граница фаз
        ax.contour(X, Y, u, levels=[0.05], colors="black", linewidths=1.2)


# --------------------------------------------------
# Общий colorbar
# --------------------------------------------------

cbar = fig.colorbar(cf, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label(
    r"Температура, $^\circ\mathrm{C}$",
    fontsize=14,
    rotation=270,
    labelpad=15,
)
cbar.set_ticks(ticks)
cbar.ax.tick_params(labelsize=10)


# --------------------------------------------------
# Сохранение
# --------------------------------------------------

plt.savefig("./graphs/melting_regimes_3x3.tif", dpi=300)
plt.show()
