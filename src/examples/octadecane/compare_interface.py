import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob
import re

from src.core.geometry import DomainGeometry
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig


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

paths = sorted(
    glob.glob("./data/checkpoint_*.npz"),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
geometry: DomainGeometry = cfg.geometry

scale = 1.0 / 884.0


def load_and_sort(path, open_curve=False):
    data = np.load(path)
    x = data["x"] * scale
    y = data["y"] * scale
    pts = np.column_stack((x, y))
    n = len(pts)
    if n <= 2:
        return x, y
    ordered = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    ordered[0] = 0
    visited[0] = True
    for i in range(1, n):
        dists = np.sum((pts[ordered[i - 1]] - pts[~visited]) ** 2, axis=1)
        next_idx = np.where(~visited)[0][np.argmin(dists)]
        ordered[i] = next_idx
        visited[next_idx] = True
    x_sorted = x[ordered]
    y_sorted = y[ordered]

    if open_curve and n > 2:
        dists = np.sqrt(np.diff(x_sorted) ** 2 + np.diff(y_sorted) ** 2)
        median_dist = np.median(dists)
        max_gap_idx = np.argmax(dists)
        if dists[max_gap_idx] > 2.0 * median_dist:
            x_sorted = np.insert(x_sorted, max_gap_idx + 1, np.nan)
            y_sorted = np.insert(y_sorted, max_gap_idx + 1, np.nan)

    return x_sorted, y_sorted


x_danaila_800, y_danaila_800 = load_and_sort(
    "./data/other_authors/danaila_800.npz", open_curve=True
)
x_okada_800, y_okada_800 = load_and_sort("./data/other_authors/okada_800.npz")
x_danaila_1575, y_danaila_1575 = load_and_sort("./data/other_authors/danaila_1575.npz")
x_okada_1575, y_okada_1575 = load_and_sort("./data/other_authors/okada_1575.npz")
x_wang_1575, y_wang_1575 = load_and_sort("./data/other_authors/wang_1575.npz")

fig, ax = plt.subplots(figsize=(8, 8))

for file_path in paths:
    data = np.load(file_path)
    u = data["u"]
    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u * cfg.delta_u + cfg.u_ref)
    x_b = np.asarray(X_b)
    y_b = np.asarray(Y_b)
    ax.plot(x_b / cfg.l, y_b / cfg.l, linestyle="--", color="red", linewidth=2.5)

ax.plot(x_danaila_800, y_danaila_800, linestyle="-", color="blue", linewidth=2.5)
ax.plot(x_danaila_1575, y_danaila_1575, linestyle="-", color="blue", linewidth=2.5)
ax.plot(x_okada_800, y_okada_800, linestyle="-", color="green", linewidth=2.5)
ax.plot(x_okada_1575, y_okada_1575, linestyle="-", color="green", linewidth=2.5)
ax.plot(x_wang_1575, y_wang_1575, linestyle="-", color="purple", linewidth=2.5)

legend_elements = [
    mlines.Line2D(
        [], [], linestyle="--", color="red", linewidth=2.5, label="Present work"
    ),
    mlines.Line2D(
        [],
        [],
        linestyle="-",
        color="blue",
        linewidth=2.5,
        label="Danaila et al. (2019)",
    ),
    mlines.Line2D(
        [], [], linestyle="-", color="green", linewidth=2.5, label="Okada (1984)"
    ),
    mlines.Line2D(
        [], [], linestyle="-", color="purple", linewidth=2.5, label="Wang et al. (2010)"
    ),
]

ax.legend(handles=legend_elements, fontsize=12)

ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("./graphs/compared.png", dpi=300)
