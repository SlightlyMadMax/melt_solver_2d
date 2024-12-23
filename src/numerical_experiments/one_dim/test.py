import numpy as np
import os

from compare_boundary import compare_num_with_analytic
from src.boundary_conditions import BoundaryCondition, BoundaryConditionType
from src.constants import ABS_ZERO
from src.geometry import DomainGeometry
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.utils import TemperatureUnit
from src.numerical_experiments.one_dim.analytic_solution_1d_2ph import (
    get_analytic_solution,
)

from src.heat_transfer.parameters import ThermalParameters
from src.heat_transfer.solver import HeatTransferSolver, HeatTransferSchemeName

if __name__ == "__main__":
    # dir_name = input("Enter a directory name where the data will be stored: ")
    dir_name = "douglas_rachford"
    dir_path = f"./results/{dir_name}"

    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass

    geometry = DomainGeometry(
        width=1.0,
        height=8.0,
        end_time=60.0 * 60.0 * 24.0 * 300.0,  # 300 days
        n_x=21,
        n_y=1001,
        n_t=7200,
    )

    print(geometry)

    max_temp = 278.15
    min_temp = 268.15
    reference_temperature = 0.5 * (min_temp + max_temp)

    thermal_params = ThermalParameters(
        u_pt=273.15,
        u_ref=reference_temperature,
        specific_heat_liquid=4120.7,
        specific_heat_solid=2056.8,
        specific_latent_heat_solid=333000.0,
        density_liquid=999.84,
        density_solid=918.9,
        thermal_conductivity_liquid=0.59,
        thermal_conductivity_solid=2.21,
    )

    print(thermal_params)

    top_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: (max_temp - thermal_params.u_ref)
        * np.ones(geometry.n_x),
    )
    right_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.NEUMANN,
        n=geometry.n_y,
        flux_func=lambda t, n: np.zeros(geometry.n_y),
    )
    bottom_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: (min_temp - thermal_params.u_ref)
        * np.ones(geometry.n_x),
    )
    left_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.NEUMANN,
        n=geometry.n_y,
        flux_func=lambda t, n: np.zeros(geometry.n_y),
    )

    heat_transfer_solver = HeatTransferSolver(
        scheme=HeatTransferSchemeName.LOC_ONE_DIM,
        geometry=geometry,
        parameters=thermal_params,
        top_bc=top_bc,
        right_bc=right_bc,
        bottom_bc=bottom_bc,
        left_bc=left_bc,
        fixed_delta=False,
        implicit_lin_max_iters=5,
        implicit_lin_stopping_criteria=1e-6,
        implicit_lin_urf=0.5,
    )

    # s_0 = float(input("Enter the initial position of the free boundary (in meters): "))

    s_0 = 0.3

    # delta = input("Enter the smoothing parameter delta or just press 'Enter' to use an adaptive one: ")

    delta = ""

    if delta == "":
        delta = None
        fixed_delta = False
    else:
        delta = float(delta)
        fixed_delta = True

    u = (
        get_analytic_solution(
            s_0=s_0,
            min_temp=min_temp,
            max_temp=max_temp,
            geometry=geometry,
            params=thermal_params,
        )
        - ABS_ZERO
        - thermal_params.u_ref
    )

    boundary = [s_0]
    times = [0.0]
    i = int(geometry.n_x / 2)

    for n in range(1, geometry.n_t):
        t = n * geometry.dt
        u = heat_transfer_solver.solve(u=u, sf=np.zeros(u.shape), time=t)
        if n % 24 == 0:
            times.append(t)
            print(f"ДЕНЬ: {int(n / 24)}")
            for j in range(geometry.n_y - 1):
                if (u[j, i] - thermal_params.u_pt_ref) * (
                    u[j + 1, i] - thermal_params.u_pt_ref
                ) < 0.0:
                    y_0 = (
                        j * geometry.dy
                        + (
                            (thermal_params.u_pt_ref - u[j, i])
                            / (u[j + 1, i] - u[j, i])
                        )
                        * geometry.dy
                    )
                    boundary.append(y_0)
                    break

    # np.savez_compressed(f"{dir_path}/1d_2f_boundary", boundary=boundary)

    compare_num_with_analytic(
        num=boundary,
        s_0=s_0,
        min_temp=min_temp,
        max_temp=max_temp,
        params=thermal_params,
        show_graphs=True,
        dir_name=dir_path,
    )
