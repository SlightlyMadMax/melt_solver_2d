import time

import numpy as np

from src.boundary_conditions import BoundaryCondition, BoundaryConditionType
from src.constants import ABS_ZERO
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.plotting import plot_velocity_field
from src.fluid_dynamics.utils import calculate_velocity_field
from src.fluid_dynamics.schemes.solver import NavierStokesSolver, NavierStokesSchemeName
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
from src.heat_transfer.schemes.solver import HeatTransferSolver, HeatTransferSchemeName


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

    min_temp = 274.15
    max_temp = 283.15
    reference_temperature = 0.5 * (min_temp + max_temp)
    # reference_temperature = max_temp

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
        delta=0.3,
    )

    print(thermal_params)

    fluid_params = FluidParameters(
        u_pt=273.15,
        u_ref=reference_temperature,
        epsilon=100000.0,
    )

    print(fluid_params)

    u = init_temperature(
        geom=geometry,
        reference_temperature=reference_temperature,
        shape=DomainShape.UNIFORM_LIQUID,
        liquid_temp=min_temp,
    )

    u[:, geometry.n_x - 1] = (max_temp - thermal_params.u_ref) * np.ones(geometry.n_y)

    print(
        f"Delta for the initial temperature distribution: {
            get_max_delta(
                u, 
                u_pt_ref=thermal_params.u_pt_ref,
            )
        }"
    )

    plot_temperature(
        u=u + thermal_params.u_ref,
        u_pt=thermal_params.u_pt,
        geom=geometry,
        time=0.0,
        graph_id=0,
        plot_boundary=False,
        show_graph=True,
        min_temp=min_temp + ABS_ZERO,
        max_temp=max_temp + ABS_ZERO,
        invert_yaxis=False,
        actual_temp_units=TemperatureUnit.KELVIN,
        display_temp_units=TemperatureUnit.CELSIUS,
    )

    # Temperature boundary conditions
    u_top_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: (min_temp - thermal_params.u_ref)
        * np.ones(geometry.n_x),
    )
    u_right_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_y,
        value_func=lambda t, n: (max_temp - thermal_params.u_ref)
        * np.ones(geometry.n_y),
    )
    u_bottom_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: (min_temp - thermal_params.u_ref)
        * np.ones(geometry.n_x),
    )
    u_left_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_y,
        value_func=lambda t, n: (min_temp - thermal_params.u_ref)
        * np.ones(geometry.n_y),
    )

    # Stream function boundary conditions
    sf_top_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: np.zeros(geometry.n_x),
    )
    sf_right_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_y,
        value_func=lambda t, n: np.zeros(geometry.n_y),
    )
    sf_bottom_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_x,
        value_func=lambda t, n: np.zeros(geometry.n_x),
    )
    sf_left_bc = BoundaryCondition(
        boundary_type=BoundaryConditionType.DIRICHLET,
        n=geometry.n_y,
        value_func=lambda t, n: np.zeros(geometry.n_y),
    )

    sf = initialize_stream_function(geom=geometry)
    w = initialize_vorticity(geom=geometry)

    heat_transfer_solver = HeatTransferSolver(
        scheme=HeatTransferSchemeName.PEACEMAN_RACHFORD,
        geometry=geometry,
        parameters=thermal_params,
        top_bc=u_top_bc,
        right_bc=u_right_bc,
        bottom_bc=u_bottom_bc,
        left_bc=u_left_bc,
        fixed_delta=False,
    )
    navier_solver = NavierStokesSolver(
        scheme=NavierStokesSchemeName.EXPLICIT_UPWIND,
        geometry=geometry,
        parameters=fluid_params,
        top_bc=sf_top_bc,
        right_bc=sf_right_bc,
        bottom_bc=sf_bottom_bc,
        left_bc=sf_left_bc,
        sf_max_iters=50,
        sf_stopping_criteria=1e-6,
        implicit_sf_max_iters=1,
        implicit_sf_stopping_criteria=1e-6,
    )

    start_time = time.process_time()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt

        u = heat_transfer_solver.solve(u=u, sf=sf, time=t, iters=3)
        sf, w = navier_solver.solve(w=w, sf=sf, u=u, time=t)

        if n % 100 == 0:
            plot_temperature(
                u=u + thermal_params.u_ref,
                u_pt=thermal_params.u_pt,
                geom=geometry,
                time=t,
                graph_id=n,
                plot_boundary=True,
                show_graph=True,
                min_temp=min_temp + ABS_ZERO,
                max_temp=max_temp + ABS_ZERO,
                invert_yaxis=False,
                actual_temp_units=TemperatureUnit.KELVIN,
                display_temp_units=TemperatureUnit.CELSIUS,
            )
            # v_x, v_y = calculate_velocity_field(sf=sf, dx=geometry.dx, dy=geometry.dy)
            # plot_velocity_field(
            #     v_x=v_x,
            #     v_y=v_y,
            #     geometry=geometry,
            #     graph_id=n,
            #     show_graph=True,
            # )
            print(
                f"Modelling Time: {n * geometry.dt} s, Execution Time: {time.process_time() - start_time} s.\n"
            )
            print(
                f"Maximum temperature value: {round(np.max(u + thermal_params.u_ref + ABS_ZERO), 2)} C"
            )
            print(
                f"Minimum temperature value: {round(np.min(u  + thermal_params.u_ref + ABS_ZERO), 2)} C"
            )
            # print(f"Maximum stream function value: {round(np.max(sf), 6)}")
            # print(f"Minimum stream function value: {round(np.min(sf), 6)}")
            # print(f"Maximum vorticity value: {round(np.max(w), 2)}")
            # print(f"Minimum vorticity value: {round(np.min(w), 2)}")

    print("Creating animation...")
    create_gif_from_images(output_filename="test_animation")
