import time
import numpy as np

from src.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditionType,
    BoundaryConditions,
)
from src.constants import ABS_ZERO
from src.convective_operators import ConvectiveTermForm
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.plotting import plot_stream_function
from src.fluid_dynamics.solvers import (
    NonIterativeNavierStokersSolver,
    VorticitySolverName,
    StreamFunctionSolverName,
)
from src.fluid_dynamics.init_values import (
    initialize_stream_function,
    initialize_vorticity,
)
from src.geometry import DomainGeometry
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.parameters import ThermalParameters
from src.heat_transfer.utils import TemperatureUnit
from src.heat_transfer.coefficient_smoothing.delta import get_max_delta
from src.heat_transfer.plotting import plot_temperature, create_gif_from_images
from src.heat_transfer.solvers import HeatTransferSolver
from src.utils import get_remaining_time


if __name__ == "__main__":
    geometry = DomainGeometry(
        width=1.0,
        height=1.0,
        end_time=60.0 * 60.0 * 24.0,
        n_x=101,
        n_y=101,
        n_t=60 * 60 * 24 * 10,
    )

    print(geometry)

    min_temp = 273.05
    max_temp = 283.15
    reference_temperature = 0.5 * (min_temp + max_temp)

    thermal_params = ThermalParameters.load_from_file("./parameters/air/thermal_params_10_6.json")
    print(thermal_params)

    fluid_params = FluidParameters.load_from_file("./parameters/air/fluid_params_10_6.json")
    print(fluid_params)

    u = init_temperature(
        geom=geometry,
        thermal_parameters=thermal_params,
        shape=DomainShape.UNIFORM_SOLID,
        solid_temp=min_temp,
    )

    u[:, 0] = (max_temp - thermal_params.u_ref) / thermal_params.delta_u

    init_delta = get_max_delta(
        u * thermal_params.delta_u + thermal_params.u_ref, u_pt=thermal_params.u_pt
    )
    print(f"Delta for the initial temperature distribution: {init_delta:.2f}")

    plot_temperature(
        u=u * thermal_params.delta_u + thermal_params.u_ref,
        u_pt=thermal_params.u_pt,
        geom=geometry,
        time=0.0,
        graph_id=0,
        plot_boundary=False,
        show_graph=True,
        min_temp=min_temp + ABS_ZERO,
        max_temp=max_temp + ABS_ZERO,
        actual_temp_units=TemperatureUnit.KELVIN,
        display_temp_units=TemperatureUnit.CELSIUS,
    )

    # Temperature boundary conditions
    u_bcs = BoundaryConditions(
        top=BoundaryCondition(
            boundary_type=BoundaryConditionType.NEUMANN,
            n=geometry.n_x,
            flux_func=lambda t, n: np.zeros(n),
        ),
        right=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_y,
            value_func=lambda t, n: (min_temp - thermal_params.u_ref)
            / thermal_params.delta_u
            * np.ones(n),
        ),
        bottom=BoundaryCondition(
            boundary_type=BoundaryConditionType.NEUMANN,
            n=geometry.n_x,
            flux_func=lambda t, n: np.zeros(n),
        ),
        left=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_y,
            value_func=lambda t, n: (max_temp - thermal_params.u_ref)
            / thermal_params.delta_u
            * np.ones(n),
        ),
    )

    # Stream function boundary conditions
    sf_bcs = BoundaryConditions(
        top=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_x,
            value_func=lambda t, n: np.zeros(n),
        ),
        right=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_y,
            value_func=lambda t, n: np.zeros(n),
        ),
        bottom=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_x,
            value_func=lambda t, n: np.zeros(n),
        ),
        left=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=geometry.n_y,
            value_func=lambda t, n: np.zeros(n),
        ),
    )

    sf = initialize_stream_function(geom=geometry)
    w = initialize_vorticity(geom=geometry)

    heat_transfer_solver = HeatTransferSolver(
        geometry=geometry,
        parameters=thermal_params,
        bcs=u_bcs,
        fixed_delta=False,
        implicit_lin_max_iters=1,
        implicit_lin_stopping_criteria=1e-6,
        implicit_lin_urf=1.0,
    )

    navier_solver = NonIterativeNavierStokersSolver(
        geometry=geometry,
        parameters=fluid_params,
        sf_bcs=sf_bcs,
        sf_max_iters=geometry.n_x * geometry.n_y,
        sf_stopping_criteria=1e-6,
    )

    start_time = time.perf_counter()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt

        u = heat_transfer_solver.solve(u=u, sf=sf, time=t)
        sf, w = navier_solver.solve(w=w, sf=sf, u=u, time=t)

        if n % 3000 == 0:
            plot_temperature(
                u=u * thermal_params.delta_u + thermal_params.u_ref,
                u_pt=thermal_params.u_pt,
                geom=geometry,
                time=t,
                graph_id=n,
                plot_boundary=True,
                show_graph=False,
                min_temp=min_temp + ABS_ZERO,
                max_temp=max_temp + ABS_ZERO,
                actual_temp_units=TemperatureUnit.KELVIN,
                display_temp_units=TemperatureUnit.CELSIUS,
            )
            plot_stream_function(
                stream_function=sf / fluid_params.v,
                geometry=geometry,
                graph_id=n,
                show_graph=False,
            )
            print(
                f"Modelling Time: {n * geometry.dt} s, "
                f"Elapsed Time: {(time.perf_counter() - start_time) / 60:.2f} min., "
                f"Estimated Remaining Time: {get_remaining_time(n=n, n_t=geometry.n_t, start_time=start_time) / 60:.2f} min.\n"
            )
            print(
                f"Maximum temperature value: {round(np.max(u * thermal_params.delta_u + thermal_params.u_ref + ABS_ZERO), 2)} C"
            )
            print(
                f"Minimum temperature value: {round(np.min(u * thermal_params.delta_u + thermal_params.u_ref + ABS_ZERO), 2)} C"
            )
            # print(f"Maximum stream function value: {round(np.max(sf), 6)}")
            # print(f"Minimum stream function value: {round(np.min(sf), 6)}")
            # print(f"Maximum vorticity value: {round(np.max(w), 2)}")
            # print(f"Minimum vorticity value: {round(np.min(w), 2)}")

    print("Creating animation...")
    create_gif_from_images(output_filename="test_animation")
