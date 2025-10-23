import math
import time

import numpy as np


from src.core.constants import ABS_ZERO
from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
)
from src.heat_transfer.init_values import init_temperature_with_interface
from src.heat_transfer.plotting import plot_temperature, create_gif_from_images
from src.heat_transfer.pt_boundary import get_pt_quadratic
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.utils import TemperatureUnit
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)
from src.utils.time_utils import get_remaining_time

max_temp = 278.15
min_temp = 268.15
reference_temperature = 0.5 * (min_temp + max_temp)

geometry = DomainGeometry(
    width=1.0,
    height=1.0,
    end_time=60.0 * 60.0 * 24.0 * 250.0,
    n_x=200,
    n_y=200,
    n_t=24 * 250,
)

print(geometry)

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
    delta=None,
    epsilon=1e-6,
)
n_x, n_y = geometry.n_x, geometry.n_y
b_lim = (
    (
        material_props.thermal_conductivity_liquid
        * material_props.specific_latent_heat
        / material_props.density_solid
    )
    * (max_temp + ABS_ZERO)
    / (
        (
            material_props.thermal_conductivity_liquid
            * material_props.specific_latent_heat
            / material_props.density_solid
        )
        * (max_temp + ABS_ZERO)
        + (
            material_props.thermal_conductivity_solid
            * material_props.specific_latent_heat
            / material_props.density_solid
        )
        * abs(min_temp + ABS_ZERO)
    )
)
print(f"Theoretical terminal boundary position: {1.0 - b_lim}")


f = [
    geometry.height / 2
    - 0.2 * math.exp(-((i * geometry.dx - geometry.width / 4.0) ** 2) / 0.005)
    + 0.2 * math.exp(-((i * geometry.dx - geometry.width / 1.5) ** 2) / 0.005)
    for i in range(geometry.n_x)
]
f = np.array(f)

u = init_temperature_with_interface(
    cfg=cfg,
    liquid_temp=max_temp,
    solid_temp=min_temp,
    f=f,
    liquid_region_height=0.0,
)

plot_temperature(
    u=u * cfg.delta_u + cfg.u_ref,
    cfg=cfg,
    graph_id=0,
    plot_boundary=True,
    show_graph=True,
    min_temp=min_temp + ABS_ZERO,
    max_temp=max_temp + ABS_ZERO,
    invert_yaxis=False,
    actual_temp_units=TemperatureUnit.KELVIN,
    display_temp_units=TemperatureUnit.CELSIUS,
    directory="./results/",
)
bcs = BoundaryConditions(
    top=const_dirichlet_condition(n_x, value=(max_temp - cfg.u_ref) / cfg.delta_u),
    right=const_neumann_condition(n_y, value=0.0),
    bottom=const_dirichlet_condition(n_x, value=(min_temp - cfg.u_ref) / cfg.delta_u),
    left=const_neumann_condition(n_y, value=0.0),
)

heat_transfer_solver = HeatTransferSolver(
    cfg=cfg,
    solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
    convective_term_form=ConvectiveTermForm.NON_DIVERGENT_CENTRAL,
    bcs=bcs,
    max_iters=1,
    tolerance=1e-6,
    urf=1.0,
)

start_time = time.perf_counter()

for i in range(1, geometry.n_t + 1):
    t = i * geometry.dt
    delta = get_mushy_zone_temperature_range(
        u * cfg.delta_u + cfg.u_ref, u_pt=cfg.material_props.u_pt
    )
    u[:, :] = heat_transfer_solver.solve(u, sf=np.zeros_like(u), time=t, delta=delta)
    if i % 24 == 0:
        print(
            f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {int(i / 24)} дней, "
            f"ВРЕМЯ ВЫПОЛНЕНИЯ: {(time.perf_counter() - start_time) / 60:.2f} мин., "
            f"ОСТАЛОСЬ: {get_remaining_time(n=i, n_t=geometry.n_t, start_time=start_time) / 60:.2f} мин."
        )
        plot_temperature(
            u=u * cfg.delta_u + cfg.u_ref,
            cfg=cfg,
            graph_id=i,
            plot_boundary=True,
            show_graph=False,
            min_temp=min_temp + ABS_ZERO,
            max_temp=max_temp + ABS_ZERO,
            invert_yaxis=False,
            actual_temp_units=TemperatureUnit.KELVIN,
            display_temp_units=TemperatureUnit.CELSIUS,
            directory="./results/",
        )

u_dim = u * cfg.delta_u - cfg.u_pt_ref
center_i_index = int(geometry.n_x / 2)
for j in range(geometry.n_y - 2):
    u0, u1, u2 = (
        u_dim[j, center_i_index],
        u_dim[j + 1, center_i_index],
        u_dim[j + 2, center_i_index],
    )
    y0 = j * geometry.dy
    y1 = (j + 1) * geometry.dy
    y2 = (j + 2) * geometry.dy
    pt = get_pt_quadratic(u0, u1, u2, cfg.u_pt_ref, y0, y1, y2)
    if pt is not None:
        print(f"Calculated final location of the boundary: {pt}")
        print(
            f"Absolute error: {abs(pt - 1 + b_lim)}, relative: {round(abs(pt - 1 + b_lim) * 100 / b_lim, 2)}%"
        )
        break

print("СОЗДАНИЕ АНИМАЦИИ...")
create_gif_from_images(
    output_filename="test_animation",
    source_directory="./results/",
    output_directory="./results/",
)
