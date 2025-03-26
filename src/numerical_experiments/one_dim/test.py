import numpy as np
import os

from scipy.optimize import fsolve

from src.boundary_conditions import (
    BoundaryConditionType,
    BoundaryConditions,
    BoundaryCondition,
)
from src.constants import ABS_ZERO
from src.convective_operators import ConvectiveTermForm
from src.geometry import DomainGeometry
from src.numerical_experiments.one_dim.analytic_solution_1d_2ph import (
    get_analytic_solution,
    trans_eq,
)
from src.heat_transfer.parameters import ThermalParameters
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName


dir_name = input("Enter a directory name where the data will be stored: ")
dir_path = f"./results/{dir_name}"

try:
    os.mkdir(dir_path)
except FileExistsError:
    pass

s_0 = float(input("Enter the initial position of the free boundary (in meters): "))

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
    end_time=60.0 * 60.0 * 24.0 * 300.0,  # 300 days
    n_x=21,
    n_y=1001,
    n_t=300 * 24,
)

print(geometry)

max_temp = 278.15
min_temp = 268.15
reference_temperature = 0.5 * (min_temp + max_temp)

thermal_params = ThermalParameters(
    domain_geometry=geometry,
    u_pt=273.15,
    u_ref=reference_temperature,
    delta_u=max_temp - min_temp,
    v=0.01,
    l=8.0,
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
    solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
    geometry=geometry,
    parameters=thermal_params,
    convective_term_form=ConvectiveTermForm.NON_DIVERGENT_CENTRAL,
    bcs=bcs,
    fixed_delta=fixed_delta,
    implicit_lin_max_iters=1,
    implicit_lin_stopping_criteria=1e-6,
    implicit_lin_urf=1.0,
)

gamma = fsolve(
    lambda x: trans_eq(
        gamma=x,
        params=thermal_params,
        min_temp=min_temp + ABS_ZERO,
        max_temp=max_temp + ABS_ZERO,
    ),
    0.0002,
)[0]

t_0: float = (s_0 / gamma) ** 2

u = (
    get_analytic_solution(
        t=t_0,
        min_temp=min_temp,
        max_temp=max_temp,
        geometry=geometry,
        params=thermal_params,
    )
    - ABS_ZERO
    - thermal_params.u_ref
) / thermal_params.delta_u

boundary = [s_0]
times = [0.0]
i = int(geometry.n_x / 2)

for n in range(1, geometry.n_t):
    t = n * geometry.dt
    u = heat_transfer_solver.solve(u=u, sf=np.zeros_like(u), time=t)
    if n % 24 == 0:
        times.append(t)
        print(f"ДЕНЬ: {int(n / 24)}")
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

u_analytical = (
    get_analytic_solution(
        t=geometry.end_time,
        min_temp=min_temp,
        max_temp=max_temp,
        geometry=geometry,
        params=thermal_params,
    )
    - ABS_ZERO
    - thermal_params.u_ref
) / thermal_params.delta_u

print(np.linalg.norm(u - u_analytical))

# compare_num_with_analytic(
#     num=boundary,
#     s_0=s_0,
#     min_temp=min_temp,
#     max_temp=max_temp,
#     params=thermal_params,
#     show_graphs=True,
#     dir_name=dir_path,
# )
