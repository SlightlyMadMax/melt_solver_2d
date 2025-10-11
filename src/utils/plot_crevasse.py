import glob
import re

import numpy as np
from matplotlib import pyplot as plt

from src.core.geometry import DomainGeometry
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig

crevasse_dir = "../../data/crevasse/convection/at_*.npz"
cfg: ExperimentConfig = ExperimentConfig.load_from_file(
    "../../parameter_sets/water/crevasse.json"
)
geometry: DomainGeometry = cfg.geometry
exp_paths = sorted(
    glob.glob(crevasse_dir),
    key=lambda f: int(re.search(r"at_(\d+)", f).group(1)),
)

fig, ax = plt.subplots()
for i, file_path in enumerate(exp_paths):
    match = re.search(r"at_(\d+)\.npz", file_path)
    n = int(match.group(1))
    u = np.load(file_path)["u"]
    # plot_temperature(
    #     u=u * cfg.delta_u + cfg.u_ref,
    #     cfg=cfg,
    #     time=n * geometry.dt,
    #     graph_id=n,
    #     plot_boundary=True,
    #     show_graph=True,
    #     actual_temp_units=TemperatureUnit.KELVIN,
    #     display_temp_units=TemperatureUnit.CELSIUS,
    # )
    X_b, Y_b = get_phase_trans_boundary(cfg=cfg, u=u * cfg.delta_u + cfg.u_ref)
    plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color="k")


ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)
# plt.tight_layout()
plt.show()
