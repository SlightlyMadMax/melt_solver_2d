import time

import numpy as np
import os


from src.core.constants import ABS_ZERO
from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import (
    BoundaryConditionType,
    BoundaryConditions,
    BoundaryCondition,
)
from src.core.geometry import DomainGeometry
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.parameters.thermal import ThermalParameters
from src.tests.numerical_experiments.one_dim.analytic_solution_1d_2ph import (
    get_analytic_solution,
)
from src.tests.numerical_experiments.one_dim.compare_boundary import (
    compare_num_with_analytic,
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
    height=8.0,
    end_time=60.0 * 60.0 * 24.0 * 100.0,  # 300 days
    n_x=21,
    n_y=201,
    n_t=100 * 240,
)

print(geometry)

max_temp = 278.15
min_temp = 268.15
reference_temperature = 0.5 * (min_temp + max_temp)

thermal_params = ThermalParameters(
    domain_geometry=geometry,
    u_pt=273.15,
    u_ref=reference_temperature,
    delta_u=0.5 * (max_temp - min_temp),
    v=0.01,
    l=geometry.length_scale,
    specific_heat_liquid=4120.7,
    specific_heat_solid=2056.8,
    specific_latent_heat=333000.0,
    density_liquid=999.84,
    density_solid=918.9,
    thermal_conductivity_liquid=0.59,
    thermal_conductivity_solid=2.21,
    delta=delta,
)

print(thermal_params)

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
    solver_name=HeatTransferSolverName.DOUGLAS_RACHFORD,
    geometry=geometry,
    parameters=thermal_params,
    convective_term_form=ConvectiveTermForm.NON_DIVERGENT_CENTRAL,
    bcs=bcs,
    fixed_delta=fixed_delta,
    max_iters=1000,
    tolerance=1e-6,
    urf=0.8,
)

u = np.ones((geometry.n_y, geometry.n_x)) * max_temp
u[0, :] = min_temp
u = (u - thermal_params.u_ref) / thermal_params.delta_u

boundary = [0.0]
time_arr = [0.0]
i = int(geometry.n_x / 2)

start_time = time.perf_counter()
for n in range(1, geometry.n_t):
    t = n * geometry.dt
    u = heat_transfer_solver.solve(u=u, sf=np.zeros_like(u), time=t)
    if n % 240 == 0:
        time_arr.append(t)
        print(f"ДЕНЬ: {int(n / 240)}")
        for j in range(geometry.n_y - 1):
            if (u[j, i] - thermal_params.u_pt_ref) * (
                u[j + 1, i] - thermal_params.u_pt_ref
            ) < 0.0:
                y_0 = (
                    j * geometry.dy
                    + ((thermal_params.u_pt_ref - u[j, i]) / (u[j + 1, i] - u[j, i]))
                    * geometry.dy
                )
                boundary.append(y_0)
                break

print(f"Elapsed Time: {time.perf_counter() - start_time:.2f} s., ")

u_analytical = (
    get_analytic_solution(
        t=(geometry.n_t - 1) * geometry.dt,
        min_temp=min_temp,
        max_temp=max_temp,
        geometry=geometry,
        params=thermal_params,
    )
    - ABS_ZERO
    - thermal_params.u_ref
) / thermal_params.delta_u

temp_top = (
    u_analytical[-1, int(geometry.n_x / 2)] * thermal_params.delta_u
    + ABS_ZERO
    + thermal_params.u_ref
)
temp_near_top = (
    u_analytical[-2, int(geometry.n_x / 2)] * thermal_params.delta_u
    + ABS_ZERO
    + thermal_params.u_ref
)
print(f"Temperature at and near the top boundary: {temp_top} C, {temp_near_top} C")
L2_error = np.linalg.norm(u[1:-1, 1:-1] - u_analytical[1:-1, 1:-1]) / np.sqrt(u[1:-1, 1:-1].size)
print(f"L2 error: {L2_error}")

compare_num_with_analytic(
    num=boundary,
    time=time_arr,
    min_temp=min_temp,
    max_temp=max_temp,
    params=thermal_params,
    show_graphs=True,
    dir_name=dir_path,
)
