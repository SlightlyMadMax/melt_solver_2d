import glob
import re

import numpy as np

from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.pt_boundary import get_pt_quadratic
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig

cfg: ExperimentConfig = ExperimentConfig.load_from_file("../../parameter_sets/water/horizontal_layer.json")
geometry: DomainGeometry = cfg.geometry

# u = np.load("../../data/wavy_surface/5x20_3/checkpoint_86400.npz")["u"]
# plot_temperature(
#     u=u * cfg.delta_u + cfg.u_ref,
#     cfg=cfg,
#     graph_id=0,
#     plot_boundary=True,
#     show_graph=False,
#     min_temp=-5,
#     max_temp=5,
#     actual_temp_units=TemperatureUnit.KELVIN,
#     display_temp_units=TemperatureUnit.CELSIUS,
#     directory="../../graphs/temperature/",
# )

files_path_mask = "../../data/wavy_surface/freezing/checkpoint_*.npz"
exp_paths = sorted(
    glob.glob(files_path_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

pt_arr = [0.0]
for file_path in exp_paths:
    match = re.search(r"checkpoint_(\d+)\.npz", file_path)
    n = int(match.group(1))
    u = np.load(file_path)["u"]
    dy = geometry.dy
    u_pt = cfg.u_pt_nd
    diff = u - u_pt
    j = np.where(diff[:-1] * diff[1:] < 0)[0][0]

    y_arr = []
    for i in range(geometry.n_x):
        j = np.where(diff[:-1] * diff[1:] < 0)[0][0]
        u0, up1 = (u[j, i], u[j + 1, i])
        y0 = j * dy
        s_lin = y0 + dy * (u_pt - u0) / (up1 - u0)
        y_arr.append(s_lin)

    pt = np.mean(np.asarray(y_arr))
    print(n, pt)
    pt_arr.append(pt)

np.savez("../../data/wavy_surface/boundary/freezing/convection_boundary.npz", b=np.asarray(pt_arr))
