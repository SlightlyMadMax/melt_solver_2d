import glob
import re

import numpy as np

from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.thermal import ThermalParameters


def plot_temperature_from_npz(
    thermal_params: ThermalParameters,
    geometry: DomainGeometry,
    min_temp: float,
    max_temp: float,
    files_path_mask: str,
):
    exp_paths = sorted(
        glob.glob(files_path_mask),
        key=lambda f: int(re.search(r"u_(\d+)", f).group(1)),
    )
    for file_path in exp_paths:
        match = re.search(r"u_(\d+)\.npz", file_path)
        n = int(match.group(1))
        u = np.load(file_path)["u"]
        plot_temperature(
            u=u * thermal_params.delta_u + thermal_params.u_ref,
            u_pt=thermal_params.u_pt,
            geometry=geometry,
            time=n * geometry.dt,
            graph_id=n,
            plot_boundary=True,
            show_graph=False,
            min_temp=min_temp + ABS_ZERO,
            max_temp=max_temp + ABS_ZERO,
            actual_temp_units=TemperatureUnit.KELVIN,
            display_temp_units=TemperatureUnit.CELSIUS,
        )
