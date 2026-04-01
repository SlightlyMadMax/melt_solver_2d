import glob
import re

import numpy as np
from matplotlib import pyplot as plt

from src.parameters.config import ExperimentConfig

solid_fraction_history = []
times = []

files_path_mask= "./data/run/checkpoint_*.npz"
exp_paths = sorted(
    glob.glob(files_path_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")

for file_path in exp_paths:
    match = re.search(r"checkpoint_(\d+)\.npz", file_path)
    n = int(match.group(1))
    u = np.load(file_path)["u"]
    solid_fraction_history.append(np.mean(u < cfg.u_pt_nd))
    times.append(n * cfg.geometry.dt)

dim_times = [
    t * cfg.stefan_number * cfg.thermal_diffusivity_ref / cfg.l**2 for t in times
]
plt.figure(figsize=(8, 5))
plt.plot(dim_times, solid_fraction_history, linewidth=1.5, label=r"$f_s(t)$")

plt.xlabel(r"Безразмерное время $\tau = Fo \cdot Ste$")
plt.ylabel(r"Доля льда")
plt.title("Изменение доли льда")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./graphs/solid_fraction.png", dpi=300)