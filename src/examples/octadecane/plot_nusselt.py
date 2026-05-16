import matplotlib.pyplot as plt
import numpy as np

from src.examples.octadecane.nusselt_correlation import nusselt_correlation
from src.parameters.config import ExperimentConfig

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
nu_history = np.load("./data/nusselt.npz")["nu"]

# Распаковка данных
times, nu_values = zip(*nu_history)

dim_times = [
    t * cfg.stefan_number * cfg.thermal_diffusivity_ref / cfg.l**2 for t in times
]
nu_pred = nusselt_correlation(dim_times, Ra=cfg.rayleigh_number)

plt.figure(figsize=(8, 5))
plt.plot(dim_times[1000:], nu_values[1000:], linewidth=1.5, label="Nu(t)")
plt.plot(dim_times[1000:], nu_pred[1000:], linewidth=1.5, label="Prediction")

plt.xlabel(r"Безразмерное время $\tau = Fo \cdot Ste$")
plt.ylabel(r"Среднее число Нуссельта $\overline{Nu}$")
plt.title("Эволюция теплопередачи на горячей стенке")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("./graphs/nusselt_evolution.png", dpi=300)
