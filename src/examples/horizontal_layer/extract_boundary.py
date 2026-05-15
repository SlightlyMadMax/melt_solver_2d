import glob
import re

import numpy as np

from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.pt_boundary import get_pt_quadratic
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig

cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
geometry: DomainGeometry = cfg.geometry

files_path_mask = "./data/melting/checkpoint_*.npz"
exp_paths = sorted(
    glob.glob(files_path_mask),
    key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
)

pt_arr = [0.05]
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

np.savez("./data/boundary/melting/convection_boundary.npz", b=np.asarray(pt_arr))
