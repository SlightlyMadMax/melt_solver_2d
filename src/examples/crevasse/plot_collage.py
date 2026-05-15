import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from src.parameters.config import ExperimentConfig
from src.core.constants import ABS_ZERO

# ── Matplotlib style ──────────────────────────────────────────────────────────
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

# ── Load Configs ──────────────────────────────────────────────────────────────
cfg_cond = ExperimentConfig.load_from_file("./conduction/config.json")
cfg_conv = ExperimentConfig.load_from_file("./convection/config.json")

CASES = {
    "Conduction": {
        "cfg": cfg_cond,
        "folder": "./data/conduction",
        "checkpoints": [600, 14400, 36000, 86400],
    },
    "Convection": {
        "cfg": cfg_conv,
        "folder": "./data/convection/colder_bottom",
        "checkpoints": [12000, 288000, 720000, 1728000],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_u_celsius(folder: str, checkpoint: int, cfg: ExperimentConfig) -> np.ndarray:
    path = Path(folder) / f"checkpoint_{checkpoint}.npz"
    u = np.load(path)["u"]
    u_dim = u * cfg.delta_u + cfg.u_ref
    return u_dim + ABS_ZERO


def find_clim(*arrays: np.ndarray):
    return min(a.min() for a in arrays), max(a.max() for a in arrays)


# ── Load data ─────────────────────────────────────────────────────────────────
data = {}
for label, cfg_case in CASES.items():
    data[label] = [
        load_u_celsius(cfg_case["folder"], ck, cfg_case["cfg"])
        for ck in cfg_case["checkpoints"]
    ]

all_arrays = [arr for arrays in data.values() for arr in arrays]
vmin, vmax = find_clim(*all_arrays)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(16, 7),
    constrained_layout=True,
)
cmap = "Blues"

for row_idx, (label, cfg_case) in enumerate(CASES.items()):
    cfg = cfg_case["cfg"]
    X, Y = cfg.geometry.mesh_grid
    arrays = data[label]

    for col_idx, (arr, step) in enumerate(zip(arrays, cfg_case["checkpoints"])):
        ax = axes[row_idx, col_idx]

        im = ax.pcolormesh(
            X,
            Y,
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

        # Изотерма T = 0 °C
        ax.contour(X, Y, arr, levels=[0.0], colors="black", linewidths=0.8)

        if col_idx > 0:
            ax.set_yticklabels([])

        axis_ticks = [0.00, 0.05, 0.10, 0.15, 0.20]
        ax.xaxis.set_major_locator(ticker.FixedLocator(axis_ticks))
        ax.yaxis.set_major_locator(ticker.FixedLocator(axis_ticks))

        # ── Формирование подписи времени ──
        t_sec = cfg.geometry.dt * step  # Физическое время в секундах

        if t_sec >= 3600:
            time_val = int(round(t_sec / 3600))
            time_str = rf"$t = {time_val}$ h"
        elif t_sec >= 60:
            time_val = int(round(t_sec / 60))
            time_str = rf"$t = {time_val}$ min"
        else:
            time_val = int(round(t_sec))
            time_str = rf"$t = {time_val}$ s"

        ax.set_title(time_str, fontsize=13, pad=6)

cbar_ticks = np.linspace(vmin, vmax, 6)
cbar = fig.colorbar(
    im,
    ax=axes,
    orientation="vertical",
    fraction=0.02,
    pad=0.02,
    label="Temperature, °C",
    ticks=cbar_ticks,
)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

fig.savefig("./graphs/collage.jpg", dpi=300, bbox_inches="tight")
plt.show()
