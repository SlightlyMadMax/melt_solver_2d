import glob
import re

import numpy as np

from src.core.constants import ABS_ZERO
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig


cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")

files_path_mask = "../data/convection/colder_bottom_continue/checkpoint_*.npz"
min_temp = 263.15
max_temp = 278.15

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
        directory="../graphs/convection/temperature_colder_continue/",
    )

# from src.heat_transfer.plotting import create_gif_from_images
#
# create_gif_from_images(
#     output_filename="crevasse_continue",
#     source_directory="./graphs/temperature_colder/",
#     output_directory="./graphs/animations/",
#     duration=200,
# )
