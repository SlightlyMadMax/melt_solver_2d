import time

import numpy as np
import os

from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import BoundaryConditions
from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.heat_transfer.coefficient_smoothing.coefficients import DeltaScheme, StepScheme
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
)
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)
from tests.numerical_experiments.one_dim.analytic_solution_1d_2ph import (
    calculate_gamma,
    get_analytical_solution,
)
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

s_0 = 0.0
t_init = 0.0

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


geometry = DomainGeometry(
    width=1.0,
    height=2.0,
    end_time=60.0 * 60.0 * 24.0,
    n_x=6,
    n_y=501,
    n_t=6 * 60 * 24,
)
n_x, n_y = geometry.n_x, geometry.n_y

cfg = ExperimentConfig(
    geometry=geometry,
    material_props=material_props,
    u_ref=0.5 * (min_temp + max_temp),
    delta_u=0.5 * (max_temp - min_temp),
    v=0.01,
    l=geometry.max_dimension,
    delta=delta,
    epsilon=1e-6,
)
gamma, residual = calculate_gamma(cfg=cfg, min_temp=min_temp, max_temp=max_temp)
print(f"Gamma: {gamma}, residual: {residual}.\n")
print(cfg)

bcs = BoundaryConditions(
    top=const_dirichlet_condition(n_x, value=(max_temp - cfg.u_ref) / cfg.delta_u),
    right=const_neumann_condition(n_y, value=0.0),
    bottom=const_dirichlet_condition(n_x, value=(min_temp - cfg.u_ref) / cfg.delta_u),
    left=const_neumann_condition(n_y, value=0.0),
)

heat_transfer_solver = HeatTransferSolver(
    cfg=cfg,
    solver_name=HeatTransferSolverName.LOC_ONE_DIM,
    convective_term_form=ConvectiveTermForm.NON_DIVERGENT_CENTRAL,
    bcs=bcs,
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

boundary = [s_0]
time_arr = [t_init]
i = int(geometry.n_x / 2)
save_interval = int(60 * 60 / geometry.dt)

start_time = time.perf_counter()
for n in range(1, geometry.n_t + 1):
    t = n * geometry.dt

    if not fixed_delta:
        delta = get_mushy_zone_temperature_range(u, u_pt=cfg.u_pt_nd, n_nodes=1)

    u[:, :] = heat_transfer_solver.solve(u=u, sf=np.zeros_like(u), time=t, delta=delta)

    if n % save_interval == 0:
        time_arr.append(t + t_init)
        print(f"ДЕНЬ: {int(n / save_interval)}")
        dy = geometry.dy
        u_pt = cfg.u_pt_nd
        s_real = gamma * t**0.5
        diff = u - u_pt
        j = np.where(diff[:-1] * diff[1:] < 0)[0][0]
        um1, u0, up1, up2 = (
            u[j - 1, i],
            u[j, i],
            u[j + 1, i],
            u[j + 2, i],
        )

        y0 = j * dy
        yp1 = (j + 1) * dy
        yp2 = (j + 2) * dy

        s_lin = y0 + dy * (u_pt - u0) / (up1 - u0)

        m_left = (u0 - um1) / dy
        s_left_piece = y0 + (u_pt - u0) / m_left

        m_right = (up2 - up1) / dy
        s_right_piece = yp1 + (u_pt - up1) / m_right
        print(
            f"s_left = {abs(s_left_piece - s_real) * 100 / s_real}, s_right = {abs(s_right_piece - s_real) * 100 / s_real}, s_lin = {abs(s_lin - s_real) * 100 / s_real}"
        )
        if (abs(m_left) > abs(m_right)) and (y0 <= s_left_piece <= yp1):
            s_num = s_left_piece
        elif y0 <= s_right_piece <= yp1:
            s_num = s_right_piece
        else:
            s_num = s_lin  # fallback

        boundary.append(s_num)

print(f"Elapsed Time: {time.perf_counter() - start_time:.2f} s., ")

rms = calculate_and_plot_temperature_error(
    cfg=cfg,
    gamma=gamma,
    num=u,
    min_temp=min_temp,
    max_temp=max_temp,
    show_graphs=True,
    dir_name=dir_path,
    t_init=t_init,
)

print(f"Temperature rms: {rms}")
calculate_and_plot_interface_error(
    cfg=cfg,
    gamma=gamma,
    num=boundary,
    time=time_arr,
    show_graphs=True,
    dir_name=dir_path,
)
