import glob
import re

import numpy as np
from matplotlib import pyplot as plt

from src.parameters.config import ExperimentConfig


cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")


s_f_cold_start = []
s_f_warm_start = []
times = []

cold_start_mask = "data/cold_start_full/checkpoint_*.npz"
cold_start_paths = sorted(
    glob.glob(cold_start_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

warm_start_mask = "data/warm_start_full/checkpoint_*.npz"
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
plt.plot(times, s_f_cold_start, linewidth=1.5, label=r"Холодный старт")
plt.plot(times, s_f_warm_start, linewidth=1.5, label=r"Горячий старт")
plt.axvline(x=2340, linestyle='--', color='gray', alpha=0.7, label=r'$t = 2340$ с')

plt.xlabel(r"Время, с")
plt.ylabel(r"Доля льда")
plt.legend()
plt.tight_layout()
plt.savefig("./graphs/ice_fraction/cold_start_vs_warm_start.png", dpi=300)
plt.show()
