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
    glob.glob("../../data/gallium/symm/u_*.npz"),
    key=lambda f: int(re.search(r"u_(\d+)", f).group(1)),
)
img = plt.imread("../../data/gau.png")

geometry = DomainGeometry(
    width=0.0889,
    height=0.0635,
    end_time=60.0 * 60.0 * 24.0,
    n_x=151,
    n_y=111,
    n_t=60 * 60 * 24 * 20,
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
    n = int(re.search(r"u_(\d+)\.npz", file_path).group(1))
    data = np.load(file_path)
    u = data["u"]
    X_b, Y_b = get_phase_trans_boundary(
        u=u * thermal_params.delta_u + thermal_params.u_ref,
        geom=geometry,
        u_pt=thermal_params.u_ref,
    )
    ax.plot(X_b, Y_b, linestyle="--", color="red", linewidth=2)


# Create custom legend handles
legend_elements = [
    mlines.Line2D(
        [], [], linestyle="--", color="red", linewidth=2, label="Численное решение"
    ),
    mlines.Line2D(
        [],
        [],
        linestyle="-",
        color="black",
        linewidth=2,
        label="Эксперимент (C. Gau, R. Viskanta)",
    ),
]
ax.legend(handles=legend_elements, fontsize=12)

ax.set_xlabel("Ширина (м)", fontsize=14)
ax.set_ylabel("Высота (м)", fontsize=14)
ax.tick_params(labelsize=12)
time_labels = {
    0.1: "2 мин",
    0.17: "3 мин",
    0.3: "6 мин",
    0.38: "8 мин",
    0.485: "10 мин",
    0.595: "12.5 мин",
    0.71: "15 мин",
    0.78: "17 мин",
    0.88: "19 мин",
}

for x_pos, label in time_labels.items():
    ax.annotate(
        label,
        xy=(x_pos, 1.02),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
plt.show()
