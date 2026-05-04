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

# ── Config ────────────────────────────────────────────────────────────────────
cfg: ExperimentConfig = ExperimentConfig.load_from_file("./convection/config.json")

X, Y = cfg.geometry.mesh_grid  # в метрах

CASES = {
    "Conduction": {
        "folder": "./data/conduction",
        "checkpoints": [600, 15000, 33000, 165000],
    },
    "Convection": {
        "folder": "./data/convection/colder_bottom",
        "checkpoints": [12000, 300000, 1080000, 3120000],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_u_celsius(folder: str, checkpoint: int) -> np.ndarray:
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
        load_u_celsius(cfg_case["folder"], ck) for ck in cfg_case["checkpoints"]
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

for row_idx, (label, arrays) in enumerate(data.items()):
    for col_idx, arr in enumerate(arrays):
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

# Colorbar с явными тиками, включая vmin и vmax
cbar_ticks = np.linspace(vmin, vmax, 6)  # 6 равномерных меток от min до max

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

fig.savefig("./collage.tiff", dpi=300, bbox_inches="tight")
plt.show()
