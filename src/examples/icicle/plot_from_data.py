import glob
import re

import numpy as np

from src.core.constants import ABS_ZERO
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig


def plot_temperature_from_npz(
    cfg: ExperimentConfig,
    min_temp: float,
    max_temp: float,
    files_path_mask: str,
):
    exp_paths = sorted(
        glob.glob(files_path_mask),
        key=lambda f: int(re.search(r"checkpoint_(\d+)", f).group(1)),
    )

    for file_path in exp_paths:
        match = re.search(r"checkpoint_(\d+)\.npz", file_path)
        n = int(match.group(1))
        u = np.load(file_path)["u"]
        plot_temperature(
            u=u * cfg.delta_u + cfg.u_ref,
            cfg=cfg,
            graph_id=n,
            plot_boundary=True,
            show_graph=False,
            min_temp=min_temp + ABS_ZERO,
            max_temp=max_temp + ABS_ZERO,
            actual_temp_units=TemperatureUnit.KELVIN,
            display_temp_units=TemperatureUnit.CELSIUS,
            directory="./graphs/5pt6c/",
        )


cfg: ExperimentConfig = ExperimentConfig.load_from_file("./parameters/5pt6c.json")
plot_temperature_from_npz(
    cfg=cfg,
    min_temp=273.15,
    max_temp=278.75,
    files_path_mask="./data/5pt6c/checkpoint_*.npz",
)
