import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob
import re

from src.core.geometry import DomainGeometry
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig


paths = sorted(
    glob.glob("./data/checkpoint_*.npz"),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
geometry: DomainGeometry = cfg.geometry


data_danaila = np.load("./data/other_authors/danaila_2019.npz")
data_okada = np.load("./data/other_authors/okada_1984.npz")
data_wang = np.load("./data/other_authors/wang_2010.npz")

scale = 1.0 / 884.0

x_danaila = data_danaila["x"] * scale
y_danaila = data_danaila["y"] * scale

x_okada = data_okada["x"] * scale
y_okada = data_okada["y"] * scale

x_wang = data_wang["x"] * scale
y_wang = data_wang["y"] * scale

fig, ax = plt.subplots()

for file_path in paths:
    data = np.load(file_path)
    u = data["u"]

    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u * cfg.delta_u + cfg.u_ref)
    x_b = np.asarray(X_b)
    y_b = np.asarray(Y_b)

    ax.plot(x_b / cfg.l, y_b / cfg.l, linestyle="--", color="red", linewidth=2)


# данные других авторов (маркеры)
ax.scatter(x_danaila, y_danaila, s=20, marker="o", label="Danaila et al. (2019)")
ax.scatter(x_okada, y_okada, s=20, marker="s", label="Okada (1984)")
ax.scatter(x_wang, y_wang, s=20, marker="^", label="Wang et al. (2010)")


# --- оформление ---
legend_elements = [
    mlines.Line2D(
        [], [], linestyle="--", color="red", linewidth=2, label="Численное решение"
    ),
    mlines.Line2D(
        [],
        [],
        linestyle="None",
        marker="o",
        color="black",
        label="Danaila et al. (2019)",
    ),
    mlines.Line2D(
        [], [], linestyle="None", marker="s", color="black", label="Okada (1984)"
    ),
    mlines.Line2D(
        [], [], linestyle="None", marker="^", color="black", label="Wang et al. (2010)"
    ),
]

ax.legend(handles=legend_elements, fontsize=12)

ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)

plt.tight_layout()
plt.show()
