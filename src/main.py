import time
import numpy as np

from src.core.boundary_conditions import (
    BoundaryCondition,
    BoundaryConditionType,
    BoundaryConditions,
)
from src.core.constants import ABS_ZERO
from src.convective_operators import ConvectiveTermForm
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.plotting import plot_stream_function
from src.fluid_dynamics.solvers.bc_correction_solver_factory import BCCorrectionNVSolver
from src.fluid_dynamics.init_values import (
    initialize_stream_function,
    initialize_vorticity,
)
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.utils import TemperatureUnit
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
)
from src.heat_transfer.plotting import plot_temperature, create_gif_from_images
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.parameters.fluid import FluidParameters
from src.parameters.thermal import ThermalParameters
from src.utils.plot_latent_heat import plot_latent_heat_field
from src.utils.time_utils import get_remaining_time


if __name__ == "__main__":
    geometry = DomainGeometry(
        width=0.0889,
        height=0.0635,
        end_time=60.0 * 60.0 * 24.0,
        n_x=151,
        n_y=111,
        n_t=60 * 60 * 24 * 20,
    )

    print(geometry)

    min_temp = 301.45
    max_temp = 311.15
    # reference_temperature = 0.5 * (min_temp + max_temp)
    # delta_u = (max_temp - min_temp) / 2

    thermal_params = ThermalParameters.load_from_file(
        "./parameter_sets/gallium/thermal_params_6_10_5.json"
    )

    print(thermal_params)

    fluid_params = FluidParameters.load_from_file(
        "./parameter_sets/gallium/fluid_params_6_10_5.json"
    )

    print(fluid_params)

    pr = (
        fluid_params.kinematic_viscosity_at_u_ref
        / thermal_params.thermal_diffusivity_solid
    )
    print(f"Pr = {pr:.4f}\n")
    print(f"Ra = {fluid_params.grashof_number * pr:.2f}\n")

    u = init_temperature(
        geometry=geometry,
        thermal_parameters=thermal_params,
        shape=DomainShape.UNIFORM_SOLID,
        solid_temp=min_temp,
    )

    u[:, 0] = (max_temp - thermal_params.u_ref) / thermal_params.delta_u
    # u[:, -1] = (max_temp - thermal_params.u_ref) / thermal_params.delta_u

    dim_u = u * thermal_params.delta_u + thermal_params.u_ref
    init_delta = np.max(
        get_mushy_zone_temperature_range(
            u=dim_u, u_pt=thermal_params.u_pt, h_x=geometry.dx, h_y=geometry.dy
        )
    )
    print(
        f"Mushy zone width for the initial temperature distribution: {init_delta:.2f}"
    )

    plot_temperature(
        u=u * thermal_params.delta_u + thermal_params.u_ref,
        u_pt=thermal_params.u_pt,
        geometry=geometry,
        time=0.0,
        graph_id=0,
        plot_boundary=True,
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
        max_iters=1,
        tolerance=1e-6,
        urf=1.0,
        solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form=ConvectiveTermForm.UPWIND,
        bc_order=1,
    )

    # navier_solver = IterativeNavierStokesSolver(
    #     geometry=geometry,
    #     parameters=fluid_params,
    #     sf_bcs=sf_bcs,
    #     vorticity_solver_name=VorticitySolverName.PEACEMAN_RACHFORD,
    #     convective_term_form=ConvectiveTermForm.UPWIND,
    #     stream_function_solver_name=StreamFunctionSolverName.MATRIX_SWEEP,
    #     max_iters=1,
    #     tolerance=1e-10,
    #     urf=1.0,
    #     bc_order=2,
    # )

    navier_solver = BCCorrectionNVSolver(
        geometry=geometry,
        parameters=fluid_params,
        sf_bcs=sf_bcs,
        sf_max_iters=geometry.n_y * geometry.n_x,
        sf_tolerance=1e-6,
    )
    u_temp, sf_temp = np.copy(u), np.copy(sf)

    start_time = time.perf_counter()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt

        u_temp = heat_transfer_solver.solve(u=u, sf=sf, time=t)
        sf_temp, _ = navier_solver.solve(w=w, sf=sf, u=u_temp, time=t)
        u = heat_transfer_solver.solve(u=u, sf=sf_temp, time=t)
        sf, w = navier_solver.solve(w=w, sf=sf, u=u, time=t)

        if n % 2000 == 0:
            # np.savez_compressed(f"../data/experiment2/u_{n}.npz", u=u)
            #
            # d = get_mushy_zone_width(
            #     u * thermal_params.delta_u + thermal_params.u_ref,
            #     u_pt=thermal_params.u_pt,
            #     h_x=geometry.dx,
            #     h_y=geometry.dy,
            #     bruh=True,
            # )
            # plot_latent_heat_field(
            #     u=u * thermal_params.delta_u + thermal_params.u_ref,
            #     u_pt=thermal_params.u_pt,
            #     delta=d,
            #     l_solid=thermal_params.volumetric_latent_heat,
            #     geometry=geometry,
            # )
            t_min = t / 60
            if t_min in {2.0, 3.0, 6.0, 8.0, 10.0, 12.5, 15.0, 17.0, 19.0}:
                np.savez_compressed(f"../data/gallium/bruh/u_{n}.npz", u=u)
                print("bruh")

            # plot_temperature(
            #     u=u * thermal_params.delta_u + thermal_params.u_ref,
            #     u_pt=thermal_params.u_pt,
            #     geometry=geometry,
            #     time=t,
            #     graph_id=n,
            #     plot_boundary=True,
            #     show_graph=False,
            #     min_temp=min_temp + ABS_ZERO,
            #     max_temp=max_temp + ABS_ZERO,
            #     actual_temp_units=TemperatureUnit.KELVIN,
            #     display_temp_units=TemperatureUnit.CELSIUS,
            # )
            # plot_stream_function(
            #     stream_function=sf * fluid_params.v * geometry.length_scale,
            #     geometry=geometry,
            #     graph_id=n,
            #     show_graph=False,
            # )
            print(
                f"Modelling Time: {n * geometry.dt:.2f} s, "
                f"Elapsed Time: {(time.perf_counter() - start_time) / 60:.2f} min., "
                f"Estimated Remaining Time: {get_remaining_time(n=n, n_t=geometry.n_t, start_time=start_time) / 60:.2f} min."
            )
            print(
                f"Maximum temperature value: {np.max(u * thermal_params.delta_u + thermal_params.u_ref + ABS_ZERO):.2f} C"
            )
            print(
                f"Minimum temperature value: {np.min(u * thermal_params.delta_u + thermal_params.u_ref + ABS_ZERO):.2f} C"
            )
            j, i = np.unravel_index(sf.argmax(), sf.shape)
            y, x = j * geometry.dy, i * geometry.dx
            print(
                f"Maximum stream function value: {np.max(sf) * fluid_params.v * geometry.length_scale / thermal_params.thermal_diffusivity_solid}, (x, y) = {x/geometry.length_scale:.3f}, {1.0 - y/geometry.length_scale:.3f}"
            )
            # print(
            #     f"Minimum stream function value: {np.min(sf) * fluid_params.v * geometry.length_scale:.3f}"
            # )
            # print(
            #     f"Maximum vorticity value: {np.max(w) * fluid_params.v / geometry.length_scale:.3f}"
            # )
            # print(
            #     f"Minimum vorticity value: {np.min(w) * fluid_params.v / geometry.length_scale:.3f}"
            # )
            print()

    # print("Creating animation...")
    # create_gif_from_images(output_filename="exp5", duration=200)
