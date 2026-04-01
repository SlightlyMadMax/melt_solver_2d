import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob
import re

from src.core.geometry import DomainGeometry
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig

paths = sorted(
    glob.glob("./data/run/checkpoint_*.npz"),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)
img = plt.imread("./data/gau.png")

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
geometry: DomainGeometry = cfg.geometry

min_temp = 301.45
max_temp = 311.15


fig, ax = plt.subplots()
ax.imshow(img, extent=[0, geometry.width, 0, geometry.height])
for file_path in paths:
    n = int(re.search(r"checkpoint_(\d+)\.npz", file_path).group(1))
    data = np.load(file_path)
    u = data["u"]
    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u * cfg.delta_u + cfg.u_ref)
    ax.plot(X_b, Y_b, linestyle="--", color="red", linewidth=2)


legend_elements = [
    mlines.Line2D(
        [], [], linestyle="--", color="red", linewidth=2, label="Численное решение"
    ),
    mlines.Line2D(
        [],
        [],
        linestyle="-",
        color="black",
        linewidth=2,
        label="Эксперимент (C. Gau, R. Viskanta)",
    ),
]
ax.legend(handles=legend_elements, fontsize=12)

ax.set_xlabel("Ширина (м)", fontsize=14)
ax.set_ylabel("Высота (м)", fontsize=14)
ax.tick_params(labelsize=12)
time_labels = {
    0.1: "2 мин",
    0.17: "3 мин",
    0.3: "6 мин",
    0.38: "8 мин",
    0.485: "10 мин",
    0.595: "12.5 мин",
    0.71: "15 мин",
    0.78: "17 мин",
    0.88: "19 мин",
}

for x_pos, label in time_labels.items():
    ax.annotate(
        label,
        xy=(x_pos, 1.02),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
plt.show()
