import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig

# Настройка шрифтов
mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
)

ABS_ZERO = -273.15  # для перевода в °C


# -----------------------------
# helper for subfigure labels
# -----------------------------
def add_subfigure_label(ax, label):
    ax.annotate(
        label,
        xy=(0.06, 0.8),
        xycoords='axes fraction',
        ha="center",
        va="center",
        fontsize=12,
        zorder=11,
        bbox=dict(
            boxstyle='circle,pad=0.3',
            facecolor='white',
            edgecolor='black',
            linewidth=1.2
        )
    )


# Загрузка конфигурации
cfg = ExperimentConfig.load_from_file(
    "../../parameter_sets/water/horizontal_layer.json"
)

geometry = cfg.geometry

# Временные шаги
melting_steps = [75000, 432000, 800400]
freezing_steps = [4800, 108000, 800400]

# Создание фигуры с дополнительным местом для colorbar
fig, axes = plt.subplots(3, 2, figsize=(14, 8))
fig.subplots_adjust(hspace=0.15, wspace=0.2, right=0.88)

# Буквы для подписей
labels = ["а", "б", "в", "г", "д", "е"]

# Для общего colorbar
vmin, vmax = -5, 5
contours = []

# Левый столбец - таяние
for i, step in enumerate(melting_steps):
    ax = axes[i, 0]

    # Загрузка данных
    data_melting = np.load(f"../../data/wavy_surface/melting/checkpoint_{step}.npz")
    u_melting = data_melting["u"]

    # Размерная температура
    u_dim = u_melting * cfg.delta_u + cfg.u_ref

    # Построение сетки (в сантиметрах)
    n_x, n_y = u_melting.shape[1], u_melting.shape[0]
    x = np.linspace(0, geometry.width * 100, n_x)
    y = np.linspace(0, geometry.height * 100, n_y)
    X, Y = np.meshgrid(x, y)

    # Температурное поле
    temp_celsius = u_dim + ABS_ZERO

    contour = ax.contourf(
        X, Y, temp_celsius, levels=101, cmap="Blues", vmin=vmin, vmax=vmax
    )
    contours.append(contour)

    # Граница раздела фаз (переводим в см)
    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u_dim)
    X_b = [x * 100 for x in X_b]
    Y_b = [y * 100 for y in Y_b]
    ax.scatter(X_b, Y_b, s=1.0, color="black")

    # Оформление
    ax.set_xlabel(r"$x$, см")
    ax.set_ylabel(r"$y$, см")
    ax.set_xlim(0, geometry.width * 100)
    ax.set_ylim(0, geometry.height * 100)
    ax.set_aspect("equal")

    # Настройка осей
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

# Правый столбец - намерзание
for i, step in enumerate(freezing_steps):
    ax = axes[i, 1]

    # Загрузка данных
    data_freezing = np.load(f"../../data/wavy_surface/freezing/checkpoint_{step}.npz")
    u_freezing = data_freezing["u"]

    # Размерная температура
    u_dim = u_freezing * cfg.delta_u + cfg.u_ref

    # Построение сетки (в сантиметрах)
    n_x, n_y = u_freezing.shape[1], u_freezing.shape[0]
    x = np.linspace(0, geometry.width * 100, n_x)
    y = np.linspace(0, geometry.height * 100, n_y)
    X, Y = np.meshgrid(x, y)

    # Температурное поле
    temp_celsius = u_dim + ABS_ZERO

    contour = ax.contourf(
        X, Y, temp_celsius, levels=101, cmap="Blues", vmin=vmin, vmax=vmax
    )

    # Граница раздела фаз (переводим в см)
    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u_dim)
    X_b = [x * 100 for x in X_b]
    Y_b = [y * 100 for y in Y_b]
    ax.scatter(X_b, Y_b, s=1.0, color="black")

    # Оформление
    ax.set_xlabel(r"$x$, см")
    ax.set_ylabel(r"$y$, см")
    ax.set_xlim(0, geometry.width * 100)
    ax.set_ylim(0, geometry.height * 100)
    ax.set_aspect("equal")

    # Настройка осей
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

# Рисуем кружочки после того, как графики готовы
for i in range(3):
    add_subfigure_label(axes[i, 0], labels[i])      # левый столбец: а, б, в
    add_subfigure_label(axes[i, 1], labels[i + 3])

# Общий вертикальный colorbar справа
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(contours[-1], cax=cbar_ax)
cbar.set_ticks(np.linspace(-5, 5, 11))
cbar.set_label("Температура, °C")

plt.savefig("../../graphs/wavy_surface/boundary_evolution.tif", dpi=300, bbox_inches="tight")
plt.show()
