import glob
import re

import numpy as np
from matplotlib import pyplot as plt

from src.parameters.config import ExperimentConfig
from src.utils.nusselt import calculate_nusselt

nusselt_history = []
times = []

files_path_mask = "data/cold_start_no_ramp_up/checkpoint_*.npz"
exp_paths = sorted(
    glob.glob(files_path_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")

for file_path in exp_paths:
    match = re.search(r"checkpoint_(\d+)\.npz", file_path)
    n = int(match.group(1))
    u = np.load(file_path)["u"]
    nu = calculate_nusselt(u=u, cfg=cfg, wall="left")
    nusselt_history.append(nu)
    times.append(n * cfg.geometry.dt)

dim_times = [
    t * cfg.stefan_number * cfg.thermal_diffusivity_ref / cfg.l**2 for t in times
]
plt.figure(figsize=(8, 5))
plt.plot(dim_times, nusselt_history, linewidth=1.5, label="Nu(t)")

plt.xlabel(r"Безразмерное время $\tau = Fo \cdot Ste$")
plt.ylabel(r"Среднее число Нуссельта $\overline{Nu}$")
plt.title("Эволюция теплопередачи на холодной стенке")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./graphs/nusselt/nu_cold_start_no_ramp_up.png", dpi=300)
