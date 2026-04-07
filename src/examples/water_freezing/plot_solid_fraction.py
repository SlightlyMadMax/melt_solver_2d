import matplotlib as mpl
import glob
import re

import numpy as np
from matplotlib import pyplot as plt

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

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")


s_f_cold_start = []
s_f_warm_start = []
times = []

cold_start_mask = "data/cold_start_no_ramp_up/checkpoint_*.npz"
cold_start_paths = sorted(
    glob.glob(cold_start_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

warm_start_mask = "data/warm_start_ramp_up/checkpoint_*.npz"
warm_start_paths = sorted(
    glob.glob(warm_start_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)


for file_path in cold_start_paths:
    match = re.search(r"checkpoint_(\d+)\.npz", file_path)
    n = int(match.group(1))
    u = np.load(file_path)["u"]
    s_f_cold_start.append(np.mean(u < cfg.u_pt_nd))
    times.append(n * cfg.geometry.dt)

for file_path in warm_start_paths:
    match = re.search(r"checkpoint_(\d+)\.npz", file_path)
    n = int(match.group(1))
    u = np.load(file_path)["u"]
    s_f_warm_start.append(np.mean(u < cfg.u_pt_nd))

plt.figure(figsize=(8, 5))
plt.plot(times[:39], s_f_cold_start[:39], linewidth=1.5, label=r"Холодный старт")
plt.plot(times[:39], s_f_warm_start[:39], linewidth=1.5, label=r"Горячий старт")

plt.xlabel(r"Время, с")
plt.ylabel(r"Доля льда")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./graphs/ice_fraction/cold_vs_warm_start.png", dpi=300)
plt.show()
