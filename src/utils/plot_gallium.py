import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import glob
import re

from src.core.geometry import DomainGeometry
from src.parameters.fluid import FluidParameters
from src.parameters.thermal import ThermalParameters
from src.heat_transfer.pt_boundary import get_phase_trans_boundary


paths = sorted(
    glob.glob("../../data/gallium/local_delta/u_*.npz"),
    key=lambda f: int(re.search(r"u_(\d+)", f).group(1)),
)
img = plt.imread("../../data/gau.png")

geometry = DomainGeometry(
    width=0.0889,
    height=0.0635,
    end_time=60.0 * 60.0 * 24.0,
    n_x=81,
    n_y=61,
    n_t=60 * 60 * 24000,
)

min_temp = 301.45
max_temp = 311.15

thermal_params = ThermalParameters.load_from_file(
    "../parameter_sets/gallium/thermal_params_6_10_5.json"
)
fluid_params = FluidParameters.load_from_file(
    "../parameter_sets/gallium/fluid_params_6_10_5.json"
)

fig, ax = plt.subplots()
ax.imshow(img, extent=[0, geometry.width, 0, geometry.height])

for file_path in paths:
    match = re.search(r"u_(\d+)\.npz", file_path)
    if match:
        n = match.group(1)
        data = np.load(file_path)
        u = data["u"]
        X_b, Y_b = get_phase_trans_boundary(
            u=u * thermal_params.delta_u + thermal_params.u_ref,
            geom=geometry,
            u_pt=thermal_params.u_ref,
        )
        ax.plot(X_b, Y_b, linestyle="--", color="red", linewidth=2)


# Create custom legend handles
smoothed_solid = mlines.Line2D(
    [],
    [],
    linestyle="--",
    color="blue",
    linewidth=2,
    label="Размазывание в твердой фазе",
)

ax.legend(
    handles=[
        smoothed_solid,
    ]
)
plt.show()
