import matplotlib.pyplot as plt
import numpy as np

from src.examples.octadecane.nusselt_correlation import nusselt_correlation
from src.parameters.config import ExperimentConfig

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
nu_history = np.load("./data/nusselt.npz")["nu"]
exp_data = np.load("./data/exp_nusselt.npz")

times, nu_values = zip(*nu_history)

dim_times = [
    t * cfg.stefan_number * cfg.thermal_diffusivity_ref / cfg.l**2 for t in times
]
nu_pred = nusselt_correlation(dim_times, Ra=cfg.rayleigh_number)

mask = exp_data["x"] <= dim_times[-1]
exp_x = exp_data["x"][mask]
exp_y = exp_data["y"][mask]

plt.figure(figsize=(8, 5))
plt.plot(dim_times[1000:], nu_values[1000:], linewidth=2.5, label="Present work")
plt.plot(
    dim_times[1000:],
    nu_pred[1000:],
    linewidth=2.5,
    linestyle="--",
    label="Jany & Bejan (1988)",
)
plt.plot(
    exp_x,
    exp_y,
    linestyle="-",
    color="black",
    linewidth=1.5,
    label="Exp. of Okada (1984)",
)

plt.xlabel(r"Dimensionless time $\tau = Fo \cdot Ste$")
plt.ylabel(r"$Nu$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(5, 9)
plt.tight_layout()
plt.savefig("./graphs/nusselt_evolution_2.png", dpi=300)
