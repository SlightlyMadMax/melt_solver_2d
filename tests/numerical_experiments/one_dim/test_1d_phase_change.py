import time

import numpy as np
import os

from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import (
    BoundaryConditionType,
    BoundaryConditions,
    BoundaryCondition,
)
from src.core.geometry import DomainGeometry
from src.heat_transfer.coefficient_smoothing.coefficients import DeltaScheme, StepScheme
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
)
from src.heat_transfer.pt_boundary import get_pt_quadratic
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from tests.numerical_experiments.one_dim.analytic_solution_1d_2ph import calculate_gamma
from tests.numerical_experiments.one_dim.compare_solution import (
    calculate_and_plot_interface_error,
    calculate_and_plot_temperature_error,
)


dir_name = input("Enter a directory name where the data will be stored: ")
dir_path = f"./results/{dir_name}"

try:
    os.mkdir(dir_path)
except FileExistsError:
    pass

delta = input(
    "Enter the smoothing parameter delta or just press 'Enter' to use an adaptive one: "
)

if delta == "":
    delta = None
    fixed_delta = False
else:
    delta = float(delta)
    fixed_delta = True

geometry = DomainGeometry(
    width=1.0,
    height=2.0,
    end_time=60.0 * 60.0 * 24.0,
    n_x=5,
    n_y=301,
    n_t=60 * 24,
)

material_props = MaterialProperties(
    u_pt=273.15,
    specific_heat_liquid=4120.7,
    specific_heat_solid=2056.8,
    specific_latent_heat=3.33e5,
    density_liquid=999.84,
    density_solid=999.84,
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

max_temp = 278.15
min_temp = 268.15

cfg = ExperimentConfig(
    geometry=geometry,
    material_props=material_props,
    u_ref=0.5 * (min_temp + max_temp),
    delta_u=0.5 * (max_temp - min_temp),
    v=0.01,
    l=geometry.max_dimension,
    u_solid=273.15 - delta / 2,
    u_liquid=273.15 + delta / 2,
    epsilon=1e-6,
    save_interval=int(60 * 60 / geometry.dt),
)
gamma = calculate_gamma(cfg=cfg, min_temp=min_temp, max_temp=max_temp)
print(cfg)

bcs = BoundaryConditions(
    top=BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: (max_temp - cfg.u_ref) / cfg.delta_u * np.ones(n),
    ),
    right=BoundaryCondition(
        boundary_type=BoundaryConditionType.NEUMANN,
        n=geometry.n_y,
        flux_func=lambda t, n: np.zeros(n),
    ),
    bottom=BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: (min_temp - cfg.u_ref) / cfg.delta_u * np.ones(n),
    ),
    left=BoundaryCondition(
        boundary_type=BoundaryConditionType.NEUMANN,
        n=geometry.n_y,
        flux_func=lambda t, n: np.zeros(n),
    ),
)

heat_transfer_solver = HeatTransferSolver(
    cfg=cfg,
    solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
    convective_term_form=ConvectiveTermForm.NON_DIVERGENT_CENTRAL,
    bcs=bcs,
    fixed_delta=fixed_delta,
    max_iters=1,
    tolerance=1e-6,
    urf=1.0,
    step_scheme=StepScheme.ERF,
    delta_scheme=DeltaScheme.GAUSS,
)

if s_0 == 0.0:
    u = np.ones((geometry.n_y, geometry.n_x)) * max_temp
    u[0, :] = min_temp
    u = (u - cfg.u_ref) / cfg.delta_u
else:
    t_init = (s_0 / gamma) ** 2
    u = get_analytical_solution(
        cfg=cfg,
        t=t_init,
        gamma=gamma,
        min_temp=min_temp,
        max_temp=max_temp,
    )
    u = (u - ABS_ZERO - cfg.u_ref) / cfg.delta_u

boundary = [0.0]
time_arr = [0.0]
i = int(geometry.n_x / 2)

start_time = time.perf_counter()
for n in range(1, geometry.n_t + 1):
    t = n * geometry.dt
    if not fixed_delta:
        delta = get_mushy_zone_temperature_range(
            u,
            u_pt=cfg.u_pt_non_dim,
            # n_nodes=1,
            # h_x=geometry.dx,
            # h_y=geometry.dy,
            # min_delta=0.01,
            # max_delta=(max_temp - min_temp) / cfg.delta_u,
        )
    else:
        delta = None
    u = heat_transfer_solver.solve(u=u, sf=np.zeros_like(u), time=t, delta=delta)
    if n % cfg.save_interval == 0:
        time_arr.append(t)
        print(f"ДЕНЬ: {int(n / cfg.save_interval)}")
        for j in range(geometry.n_y - 2):
            u0, u1, u2 = (
                u[j, i],
                u[j + 1, i],
                u[j + 2, i],
            )
            u_ref = cfg.u_pt_ref
            y0 = j * geometry.dy
            y1 = (j + 1) * geometry.dy
            y2 = (j + 2) * geometry.dy
            pt = get_pt_quadratic(u0, u1, u2, u_ref, y0, y1, y2)
            if pt is not None:
                boundary.append(pt)
                break

print(f"Elapsed Time: {time.perf_counter() - start_time:.2f} s., ")

gamma = calculate_gamma(cfg=cfg, min_temp=min_temp, max_temp=max_temp)

calculate_and_plot_temperature_error(
    cfg=cfg,
    gamma=gamma,
    num=u,
    min_temp=min_temp,
    max_temp=max_temp,
    show_graphs=True,
    dir_name=dir_path,
)
calculate_and_plot_interface_error(
    cfg=cfg,
    gamma=gamma,
    num=boundary,
    time=time_arr,
    show_graphs=True,
    dir_name=dir_path,
)
