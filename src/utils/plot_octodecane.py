import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob
import re

from src.core.geometry import DomainGeometry
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig

paths = sorted(
    glob.glob("../../data/octodecane/test/u_*.npz"),
    key=lambda f: int(re.search(r"u_(\d+)", f).group(1)),
)
img = plt.imread("../../data/danaila.png")

cfg: ExperimentConfig = ExperimentConfig.load_from_file(
    "../../parameter_sets/octodecane/config.json"
)
geometry: DomainGeometry = cfg.geometry

min_temp = 301.2426
max_temp = 310.07


fig, ax = plt.subplots()
ax.imshow(img, extent=[0, geometry.width, 0, geometry.height])
for file_path in paths:
    n = int(re.search(r"u_(\d+)\.npz", file_path).group(1))
    data = np.load(file_path)
    u = data["u"]
    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u * cfg.delta_u + cfg.u_ref)
    ax.plot(X_b, Y_b, linestyle="--", color="red", linewidth=2)


legend_elements = [
    mlines.Line2D(
        [], [], linestyle="--", color="red", linewidth=2, label="Численное решение"
    ),
]
ax.legend(handles=legend_elements, fontsize=12)

ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)
plt.tight_layout()
plt.show()
