import time
import numpy as np

from src.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditionType,
    BoundaryConditions,
)
from src.constants import ABS_ZERO
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
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName


if __name__ == "__main__":
    geometry = DomainGeometry(
        width=1.0,
        height=1.0,
        end_time=60.0 * 60.0 * 24.0 * 7.0,
        n_x=101,
        n_y=101,
        n_t=60 * 60 * 24 * 70,
    )

    print(geometry)

    min_temp = 273.05
    max_temp = 283.15
    reference_temperature = 0.5 * (min_temp + max_temp)

    thermal_params = ThermalParameters(
        domain_geometry=geometry,
        u_pt=273.15,
        u_ref=reference_temperature,
        delta_u=abs(max_temp - reference_temperature),
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

    fluid_params = FluidParameters(
        domain_geometry=geometry,
        u_pt=273.15,
        u_ref=reference_temperature,
        delta_u=abs(max_temp - reference_temperature),
        v=0.01,
        epsilon=5e-2,
    )

    print(fluid_params)

    u = init_temperature(
        geom=geometry,
        thermal_parameters=thermal_params,
        shape=DomainShape.UNIFORM_SOLID,
        solid_temp=min_temp,
    )

    u[:, 0] = (max_temp - thermal_params.u_ref) / thermal_params.delta_u

    print(
        f"Delta for the initial temperature distribution: {
            get_max_delta(
                u * thermal_params.delta_u + thermal_params.u_ref,
                u_pt=thermal_params.u_pt,
            )
        :.2E}"
    )

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
        implicit_lin_max_iters=2,
        implicit_lin_stopping_criteria=1e-6,
        implicit_lin_urf=1.0,
    )

    navier_solver = NonIterativeNavierStokersSolver(
        geometry=geometry,
        parameters=fluid_params,
        sf_bcs=sf_bcs,
        sf_max_iters=1000,
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
                f"Modelling Time: {n * geometry.dt} s, Execution Time: {time.perf_counter() - start_time:.4f} s.\n"
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
