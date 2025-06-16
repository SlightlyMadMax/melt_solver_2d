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
from src.fluid_dynamics.utils import calculate_velocity_from_sf
from src.heat_transfer.coefficient_smoothing.coefficients import (
    StepScheme,
    DeltaScheme,
)
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.utils import TemperatureUnit
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
    get_delta,
)
from src.heat_transfer.plotting import plot_temperature, create_gif_from_images
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.parameters.fluid import FluidParameters
from src.parameters.thermal import ThermalParameters
from src.utils.stand_with_icicle import init_temperature_icicle
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

    dx, dy = geometry.dx, geometry.dy
    dt = geometry.dt
    n_x, n_y, n_t = geometry.n_x, geometry.n_y, geometry.n_t
    l = geometry.length_scale
    min_temp = 301.45
    max_temp = 311.15

    # reference_temperature = max_temp
    # delta_u = max_temp - min_temp

    thermal_params = ThermalParameters.load_from_file(
        "./parameter_sets/gallium/thermal_params_6_10_5.json"
    )

    # thermal_params = ThermalParameters(
    #     u_pt=273.15,
    #     u_ref=reference_temperature,
    #     delta_u=delta_u,
    #     v=0.01,
    #     l=l,
    #     specific_heat_liquid=4120.7,
    #     specific_heat_solid=2056.8,
    #     specific_latent_heat=333000.0,
    #     density_liquid=999.84,
    #     density_solid=918.9,
    #     thermal_conductivity_liquid=0.59,
    #     thermal_conductivity_solid=2.21,
    # )

    print(thermal_params)

    delta_u = thermal_params.delta_u
    u_ref = thermal_params.u_ref
    u_pt = thermal_params.u_pt

    fluid_params = FluidParameters.load_from_file(
        "./parameter_sets/gallium/fluid_params_6_10_5.json"
    )

    # fluid_params = FluidParameters(
    #     u_pt=273.15,
    #     u_ref=reference_temperature,
    #     delta_u=delta_u,
    #     v=0.01,
    #     l=l,
    #     epsilon=0.000001,
    #     kinematic_viscosity_coeffs=[
    #         0.000108963453,
    #         -9.28722151e-07,
    #         2.65889022e-09,
    #         -2.54761652e-12,
    #     ],
    #     volumetric_thermal_exp_coeffs=[-0.0114630054, 6.86739177e-05, -9.84848485e-08],
    # )

    print(fluid_params)

    v = fluid_params.v
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
    # u = init_temperature_icicle(
    #     geometry=geometry,
    #     thermal_parameters=thermal_params,
    #     liquid_temp=max_temp,
    #     solid_temp=min_temp,
    # )

    u[:, 0] = (max_temp - u_ref) / delta_u
    # u[:, -1] = (max_temp - u_ref) / delta_u

    dim_u = u * delta_u + u_ref
    init_delta = get_mushy_zone_temperature_range(u=dim_u, u_pt=u_pt)

    print(
        f"Mushy zone width for the initial temperature distribution: {init_delta:.2f}"
    )

    plot_temperature(
        u=dim_u,
        u_pt=u_pt,
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
            n=n_x,
            flux_func=lambda t, n: np.zeros(n),
        ),
        right=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=n_y,
            value_func=lambda t, n: (min_temp - u_ref) / delta_u * np.ones(n),
            flux_func=lambda t, n: np.zeros(n),
        ),
        bottom=BoundaryCondition(
            boundary_type=BoundaryConditionType.NEUMANN,
            n=n_x,
            flux_func=lambda t, n: np.zeros(n),
        ),
        left=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=n_y,
            value_func=lambda t, n: (max_temp - u_ref) / delta_u * np.ones(n),
            flux_func=lambda t, n: np.zeros(n),
        ),
    )

    # Stream function boundary conditions
    sf_bcs = BoundaryConditions(
        top=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=n_x,
            value_func=lambda t, n: np.zeros(n),
        ),
        right=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=n_y,
            value_func=lambda t, n: np.zeros(n),
        ),
        bottom=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=n_x,
            value_func=lambda t, n: np.zeros(n),
        ),
        left=BoundaryCondition(
            boundary_type=BoundaryConditionType.DIRICHLET,
            n=n_y,
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
        step_scheme=StepScheme.ERF,
        delta_scheme=DeltaScheme.GAUSS,
    )

    navier_solver = BCCorrectionNVSolver(
        geometry=geometry,
        parameters=fluid_params,
        sf_bcs=sf_bcs,
        sf_max_iters=n_y * n_x,
        sf_tolerance=1e-6,
        convective_term_form=ConvectiveTermForm.UPWIND,
    )

    v_x = np.zeros_like(sf)
    v_y = np.zeros_like(sf)

    start_time = time.perf_counter()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt
        # delta = get_delta(
        #     u_n_non=u,
        #     sf=sf,
        #     time=t,
        #     solver=heat_transfer_solver,
        #     params=thermal_params,
        #     geometry=geometry,
        #     delta_min=0.1,
        #     delta_max=5.0,
        #     tol=1e-3,
        # )
        delta = get_mushy_zone_temperature_range(u * delta_u + u_ref, u_pt=u_pt)
        u = heat_transfer_solver.solve(u=u, sf=sf, delta=delta, time=t)
        # delta = get_delta(
        #     u_n_non=u,
        #     sf=sf,
        #     time=t,
        #     solver=heat_transfer_solver,
        #     params=thermal_params,
        #     geometry=geometry,
        #     delta_min=0.1,
        #     delta_max=5.0,
        #     tol=1e-3,
        # )
        delta = get_mushy_zone_temperature_range(u * delta_u + u_ref, u_pt=u_pt)
        sf, w = navier_solver.solve(w=w, sf=sf, u=u, delta=delta, time=t)

        if n % 200 == 0:
            t_min = t / 60
            if t_min in {2.0, 3.0, 6.0, 8.0, 10.0, 12.5, 15.0, 17.0, 19.0}:
                print("bruh")
                np.savez_compressed(f"../data/gallium/better_courant/u_{n}.npz", u=u)
            u_dim = u * delta_u + u_ref
            calculate_velocity_from_sf(
                sf * v * l,
                v_x,
                v_y,
                geometry.dx,
                geometry.dy,
            )
            print(f"Courant number: {max(np.max(np.abs(v_x)*dt/dx), np.max(np.abs(v_y)*dt/dy))}")
            # plot_temperature(
            #     u=u_dim,
            #     u_pt=u_pt,
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
            plot_stream_function(
                stream_function=sf * v * l,
                geometry=geometry,
                graph_id=n,
                show_graph=False,
            )
            print(
                f"Modelling Time: {n * dt:.2f} s, "
                f"Elapsed Time: {(time.perf_counter() - start_time) / 60:.2f} min., "
                f"Estimated Remaining Time: {get_remaining_time(n=n, n_t=n_t, start_time=start_time) / 60:.2f} min."
            )
            print(f"Maximum temperature value: {np.max(u_dim + ABS_ZERO):.2f} C")
            print(f"Minimum temperature value: {np.min(u_dim + ABS_ZERO):.2f} C")
            # j, i = np.unravel_index(sf.argmax(), sf.shape)
            # y, x = j * dy, i * dx
            # print(
            #     f"Maximum stream function value: {np.max(sf) * v * l}, (x, y) = {x/l:.3f}, {1.0 - y/l:.3f}"
            # )
            # print(
            #     f"Minimum stream function value: {np.min(sf) * v * l:.3f}"
            # )
            # print(
            #     f"Maximum vorticity value: {np.max(w) * v / l:.3f}"
            # )
            # print(
            #     f"Minimum vorticity value: {np.min(w) * v / l:.3f}"
            # )
            print()

    print("Creating animation...")
    create_gif_from_images(output_filename="exp5", duration=200)
