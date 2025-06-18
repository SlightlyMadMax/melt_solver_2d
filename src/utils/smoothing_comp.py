import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from tests.numerical_experiments.one_dim.analytic_solution_1d_2ph import trans_eq

geometry = DomainGeometry(
    width=1.0,
    height=5.0,
    end_time=60.0 * 60.0 * 24.0 * 14.0,  # 300 days
    n_x=21,
    n_y=501,
    n_t=24 * 14 * 60,
)
max_temp = 278.15
min_temp = 268.15
reference_temperature = 0.5 * (min_temp + max_temp)

material_props = MaterialProperties(
    u_pt=273.15,
    specific_heat_liquid=4120.7,
    specific_heat_solid=2056.8,
    specific_latent_heat=3.33e5,
    density_liquid=999.84,
    density_solid=918.9,
    thermal_conductivity_liquid=0.59,
    thermal_conductivity_solid=2.21,
    kinematic_viscosity_coeffs=[
        0.000108963453,
        -9.28722151e-07,
        2.65889022e-09,
        -2.54761652e-12,
    ],
    volumetric_thermal_exp_coeffs=[7.68e-6],
)

cfg = ExperimentConfig(
    geometry=geometry,
    material_props=material_props,
    u_ref=0.5 * (min_temp + max_temp),
    delta_u=0.5 * (max_temp - min_temp),
    v=0.01,
    l=geometry.max_dimension,
    epsilon=1e-6,
    delta=0.3,
)

gamma = fsolve(
    lambda x: trans_eq(
        gamma=x,
        material_props=material_props,
        min_temp=min_temp + ABS_ZERO,
        max_temp=max_temp + ABS_ZERO,
    ),
    0.0002,
)[0]

# путь к папке с результатами (относительно вашей рабочей директории)
base_dir = "../../tests/numerical_experiments/one_dim/results"

methods = [
    ("box", "Линейная + кусочно-постоянная"),
    ("box_c_eff", "Линейная + кусочно-постоянная + EHCM"),
    ("parabolic", "Парабола + линейная"),
    ("hyper", "сosh + tanh"),
    ("gauss", "Гауссиана + erf"),
]
custom_colors = {
    "box": "C0",
    "box_c_eff": "C1",
    "parabolic": "C2",
    "hyper": "grey",
    "gauss": "crimson",  # emphasized
}
plt.figure(figsize=(8, 5))
for folder_name, label in methods:
    path = os.path.join(base_dir, folder_name, "boundary.npz")
    if not os.path.isfile(path):
        print(f"Warning: файл не найден {path}")
        continue

    data = np.load(path)
    time_arr = data["time_arr"]  # предполагается shape (n,)
    boundary_num = data["boundary"]  # предполагается shape (n,)

    # аналитическое решение
    boundary_exact = gamma * np.sqrt(time_arr)

    # абсолютная ошибка
    error = np.abs(boundary_num - boundary_exact)
    plt.plot(
        time_arr / 86400,  # переводим время в часы или дни
        error,
        label=label,
        linewidth=1.5,
        color=custom_colors[folder_name]
    )

# plt.yscale("log")
plt.xlabel("Время (дни)", fontsize=14)
plt.ylabel("Абс. погрешность $s(t)$ (м)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
