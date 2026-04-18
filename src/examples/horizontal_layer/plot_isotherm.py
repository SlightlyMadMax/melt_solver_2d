import glob
import re

import numpy as np
from matplotlib import pyplot as plt

from src.core.constants import ABS_ZERO
from src.parameters.config import ExperimentConfig


cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")

files_path_mask = "./data/freezing/checkpoint_*.npz"
exp_paths = sorted(
    glob.glob(files_path_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

min_temp = 268.15 + ABS_ZERO
max_temp = 278.15 + ABS_ZERO

X, Y = cfg.geometry.mesh_grid

# Значения изотерм, которые нужно отобразить (в °C)
isotherms_celsius = [0, 4]

for file_path in exp_paths:
    match = re.search(r"checkpoint_(\d+)\.npz", file_path)
    n = int(match.group(1))
    u = np.load(file_path)["u"]
    u_c = u * cfg.delta_u + cfg.u_ref + ABS_ZERO  # температура в °C

    levels = np.linspace(min_temp, max_temp, 101)

    ax = plt.axes(
        xlim=(0.0, cfg.geometry.width),
        ylim=(0.0, cfg.geometry.height),
        xlabel="x, м",
        ylabel="y, м",
    )
    ax.set_xlim(0, cfg.geometry.width)
    ax.set_ylim(0, cfg.geometry.height)
    ax.set_aspect("equal")

    contour = plt.contourf(
        X,
        Y,
        u_c,
        levels=levels,
        cmap="Blues",
        extend="both",
    )

    # Добавление изотерм 0°C и 4°C
    contour_lines = plt.contour(
        X,
        Y,
        u_c,
        levels=isotherms_celsius,
        colors=["blue", "red"],
        linewidths=1.5,
        linestyles=["--", "-."],
        alpha=0.9,
    )

    # Подписи значений на изотермах
    plt.clabel(contour_lines, fmt="%1.0f°C", fontsize=9, inline=True)

    cbar = plt.colorbar(contour)
    cbar.set_ticks(np.linspace(min_temp, max_temp, num=7))
    cbar.set_label("Температура, °С", rotation=270, labelpad=15)

    plt.savefig(f"./graphs/temperature/freezing/T_{n}.jpg", dpi=300)
    plt.close()
