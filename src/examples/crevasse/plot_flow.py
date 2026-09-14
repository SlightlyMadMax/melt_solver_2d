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

# ── Case ──────────────────────────────────────────────────────────────────────
cfg = ExperimentConfig.load_from_file("./convection/config.json")
FOLDER = "./data/convection/48_hrs"
CHECKPOINTS = [12000, 288000, 720000, 1728000]

# Water temperature range shown in colour; ice is drawn grey
T_WATER_MIN, T_WATER_MAX = 0.0, 5.0
# Temperature of maximum density of fresh water, drawn as a dashed isotherm
T_DENSITY_MAX = 4.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_fields(folder: str, checkpoint: int, cfg: ExperimentConfig):
    """Temperature in °C and dimensional velocity components in m/s."""
    data = np.load(Path(folder) / f"checkpoint_{checkpoint}.npz")
    t_celsius = data["u"] * cfg.delta_u + cfg.u_ref + ABS_ZERO
    return t_celsius, data["v_x"] * cfg.v, data["v_y"] * cfg.v


def time_label(t_sec: float) -> str:
    if t_sec >= 3600:
        return rf"$t = {int(round(t_sec / 3600))}$ h"
    if t_sec >= 60:
        return rf"$t = {int(round(t_sec / 60))}$ min"
    return rf"$t = {int(round(t_sec))}$ s"


# ── Plot ──────────────────────────────────────────────────────────────────────
X, Y = cfg.geometry.mesh_grid
x, y = X[0, :], Y[:, 0]

cmap = plt.get_cmap("RdYlBu_r").copy()
cmap.set_bad("0.85")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), constrained_layout=True)

for idx, (ax, step) in enumerate(zip(axes.ravel(), CHECKPOINTS)):
    t_c, v_x, v_y = load_fields(FOLDER, step, cfg)
    liquid = t_c >= 0.0
    t_water = np.ma.masked_where(~liquid, t_c)

    im = ax.pcolormesh(
        X,
        Y,
        t_water,
        cmap=cmap,
        vmin=T_WATER_MIN,
        vmax=T_WATER_MAX,
        shading="auto",
    )

    # Streamlines in the water only; line width follows the local speed
    speed = np.hypot(v_x, v_y)
    speed_max = speed[liquid].max()
    ax.streamplot(
        x,
        y,
        np.ma.masked_where(~liquid, v_x),
        np.ma.masked_where(~liquid, v_y),
        color="black",
        linewidth=0.3 + 1.8 * np.ma.masked_where(~liquid, speed) / speed_max,
        density=2.0,
        arrowsize=0.6,
    )

    # Изотерма T = 0 °C
    ax.contour(X, Y, t_c, levels=[0.0], colors="black", linewidths=1.2)
    # Изотерма максимальной плотности T = 4 °C
    ax.contour(
        X, Y, t_water, levels=[T_DENSITY_MAX], colors="white", linewidths=1.0, linestyles="--"
    )

    axis_ticks = [0.00, 0.05, 0.10, 0.15, 0.20]
    ax.xaxis.set_major_locator(ticker.FixedLocator(axis_ticks))
    ax.yaxis.set_major_locator(ticker.FixedLocator(axis_ticks))
    ax.set_xlim(0.0, cfg.geometry.width)
    ax.set_ylim(0.0, cfg.geometry.height)
    ax.set_aspect("equal")

    t_sec = cfg.geometry.dt * step
    ax.set_title(
        rf"{time_label(t_sec)}, $|\mathbf{{v}}|_\mathrm{{max}} = {speed_max * 1e3:.2f}$ mm/s",
        fontsize=13,
        pad=6,
    )

cbar = fig.colorbar(
    im,
    ax=axes,
    orientation="vertical",
    shrink=0.6,
    pad=0.02,
    label="Water temperature, °C",
    ticks=np.linspace(T_WATER_MIN, T_WATER_MAX, 6),
)
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

fig.savefig("./graphs/flow.jpg", dpi=300, bbox_inches="tight")
plt.show()
