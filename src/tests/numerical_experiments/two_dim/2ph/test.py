import math
import time

import numpy as np

from src.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditionType,
    BoundaryConditions,
)
from src.constants import ABS_ZERO
from src.convective_operators import ConvectiveTermForm
from src.geometry import DomainGeometry
from src.heat_transfer.init_values import init_temperature_with_interface
from src.heat_transfer.parameters import ThermalParameters
from src.heat_transfer.plotting import plot_temperature, create_gif_from_images
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.utils import TemperatureUnit
from src.utils import get_remaining_time

max_temp = 278.15
min_temp = 268.15
reference_temperature = 0.5 * (min_temp + max_temp)

geometry = DomainGeometry(
    width=1.0,
    height=1.0,
    end_time=60.0 * 60.0 * 24.0 * 250.0,
    n_x=100,
    n_y=100,
    n_t=24 * 250 * 60,
)

print(geometry)

thermal_params = ThermalParameters(
    domain_geometry=geometry,
    u_pt=273.15,
    u_ref=reference_temperature,
    delta_u=max_temp - min_temp,
    v=0.01,
    specific_heat_liquid=4120.7,
    specific_heat_solid=2056.8,
    specific_latent_heat_solid=333000.0,
    density_liquid=999.84,
    density_solid=918.9,
    thermal_conductivity_liquid=0.59,
    thermal_conductivity_solid=2.21,
)

print(thermal_params)

b_lim = (
    (
        thermal_params.thermal_conductivity_liquid
        * thermal_params.specific_latent_heat
        / thermal_params.density_solid
    )
    * (max_temp + ABS_ZERO)
    / (
        (
            thermal_params.thermal_conductivity_liquid
            * thermal_params.specific_latent_heat
            / thermal_params.density_solid
        )
        * (max_temp + ABS_ZERO)
        + (
            thermal_params.thermal_conductivity_solid
            * thermal_params.specific_latent_heat
            / thermal_params.density_solid
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
    geom=geometry,
    thermal_parameters=thermal_params,
    liquid_temp=max_temp,
    solid_temp=min_temp,
    f=f,
    liquid_region_height=0.0,
)

plot_temperature(
    u=u * thermal_params.delta_u + thermal_params.u_ref,
    u_pt=thermal_params.u_pt,
    geom=geometry,
    time=0.0,
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
    top=BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: (max_temp - thermal_params.u_ref)
        / thermal_params.delta_u
        * np.ones(n),
    ),
    right=BoundaryCondition(
        boundary_type=BoundaryConditionType.NEUMANN,
        n=geometry.n_y,
        flux_func=lambda t, n: np.zeros(n),
    ),
    bottom=BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: (min_temp - thermal_params.u_ref)
        / thermal_params.delta_u
        * np.ones(n),
    ),
    left=BoundaryCondition(
        boundary_type=BoundaryConditionType.NEUMANN,
        n=geometry.n_y,
        flux_func=lambda t, n: np.zeros(n),
    ),
)

heat_transfer_solver = HeatTransferSolver(
    solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
    geometry=geometry,
    parameters=thermal_params,
    convective_term_form=ConvectiveTermForm.NON_DIVERGENT_CENTRAL,
    bcs=bcs,
    fixed_delta=False,
    max_iters=1,
    tolerance=1e-6,
    urf=1.0,
)

start_time = time.perf_counter()

for i in range(1, geometry.n_t + 1):
    t = i * geometry.dt
    u = heat_transfer_solver.solve(u, sf=np.zeros_like(u), time=t)
    if i % (24 * 60) == 0:
        print(
            f"ВРЕМЯ МОДЕЛИРОВАНИЯ: {i / 60} ч, "
            f"ВРЕМЯ ВЫПОЛНЕНИЯ: {(time.perf_counter() - start_time) / 60:.2f} мин., "
            f"ОСТАЛОСЬ: {get_remaining_time(n=i, n_t=geometry.n_t, start_time=start_time) / 60:.2f} мин."
        )
        plot_temperature(
            u=u * thermal_params.delta_u + thermal_params.u_ref,
            u_pt=thermal_params.u_pt,
            geom=geometry,
            time=t,
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

for j in range(1, geometry.n_y - 1):
    if (
        u[j, int(geometry.n_x / 2)] * thermal_params.delta_u - thermal_params.u_pt_ref
    ) * (
        u[j + 1, int(geometry.n_x / 2)] * thermal_params.delta_u
        - thermal_params.u_pt_ref
    ) < 0.0:
        y_0 = abs(
            (
                u[j, int(geometry.n_x / 2)] * (j + 1) * geometry.dy
                - u[j + 1, int(geometry.n_x / 2)] * j * geometry.dy
            )
            / (u[j, int(geometry.n_x / 2)] - u[j + 1, int(geometry.n_x / 2)])
        )
        print(f"Calculated final location of the boundary: {y_0}")
        print(
            f"Absolute error: {abs(y_0 - 1 + b_lim)}, relative: {round(abs(y_0 - 1 + b_lim) * 100 / b_lim, 2)}%"
        )
        break

print("СОЗДАНИЕ АНИМАЦИИ...")
create_gif_from_images(
    output_filename="test_animation",
    source_directory="./results/",
    output_directory="./results/",
)
