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
from src.heat_transfer.coefficient_smoothing.coefficients import (
    StepScheme,
    DeltaScheme,
    get_delta_fn,
)
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.utils import TemperatureUnit
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
    find_best_delta,
)
from src.heat_transfer.plotting import plot_temperature, create_gif_from_images
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.parameters.fluid import FluidParameters
from src.parameters.thermal import ThermalParameters
from src.utils.stand_with_icicle import init_temperature_icicle
from src.utils.time_utils import get_remaining_time


if __name__ == "__main__":
    geometry = DomainGeometry(
        width=0.4,
        height=0.2,
        end_time=60.0 * 60.0 * 24.0,
        n_x=201,
        n_y=101,
        n_t=60 * 60 * 24 * 100,
    )

    print(geometry)

    min_temp = 273.14
    max_temp = 277.15
    reference_temperature = max_temp
    delta_u = max_temp - min_temp

    # thermal_params = ThermalParameters.load_from_file(
    #     "./parameter_sets/gallium/thermal_params_6_10_5.json"
    # )

    thermal_params = ThermalParameters(
        u_pt=273.15,
        u_ref=reference_temperature,
        delta_u=delta_u,
        v=0.01,
        l=geometry.length_scale,
        specific_heat_liquid=4120.7,
        specific_heat_solid=2056.8,
        specific_latent_heat=333000.0,
        density_liquid=999.84,
        density_solid=918.9,
        thermal_conductivity_liquid=0.59,
        thermal_conductivity_solid=2.21,
    )

    print(thermal_params)

    # fluid_params = FluidParameters.load_from_file(
    #     "./parameter_sets/gallium/fluid_params_6_10_5.json"
    # )

    fluid_params = FluidParameters(
        u_pt=273.15,
        u_ref=reference_temperature,
        delta_u=delta_u,
        v=0.01,
        l=geometry.length_scale,
        epsilon=0.000001,
        kinematic_viscosity_coeffs=[
            0.000108963453,
            -9.28722151e-07,
            2.65889022e-09,
            -2.54761652e-12,
        ],
        volumetric_thermal_exp_coeffs=[-0.0114630054, 6.86739177e-05, -9.84848485e-08],
    )

    print(fluid_params)

    pr = (
        fluid_params.kinematic_viscosity_at_u_ref
        / thermal_params.thermal_diffusivity_solid
    )
    print(f"Pr = {pr:.4f}\n")
    print(f"Ra = {fluid_params.grashof_number * pr:.2f}\n")

    # u = init_temperature(
    #     geometry=geometry,
    #     thermal_parameters=thermal_params,
    #     shape=DomainShape.UNIFORM_SOLID,
    #     solid_temp=min_temp,
    # )
    u = init_temperature_icicle(
        geometry=geometry,
        thermal_parameters=thermal_params,
        liquid_temp=max_temp,
        solid_temp=min_temp,
    )

    # u[:, 0] = (max_temp - thermal_params.u_ref) / thermal_params.delta_u
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
        u=dim_u,
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
            boundary_type=BoundaryConditionType.NEUMANN,
            n=geometry.n_y,
            value_func=lambda t, n: (min_temp - thermal_params.u_ref)
            / thermal_params.delta_u
            * np.ones(n),
            flux_func=lambda t, n: np.zeros(n),
        ),
        bottom=BoundaryCondition(
            boundary_type=BoundaryConditionType.NEUMANN,
            n=geometry.n_x,
            flux_func=lambda t, n: np.zeros(n),
        ),
        left=BoundaryCondition(
            boundary_type=BoundaryConditionType.NEUMANN,
            n=geometry.n_y,
            value_func=lambda t, n: (max_temp - thermal_params.u_ref)
            / thermal_params.delta_u
            * np.ones(n),
            flux_func=lambda t, n: np.zeros(n),
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
        step_scheme=StepScheme.ERF,
        delta_scheme=DeltaScheme.GAUSS,
    )

    navier_solver = BCCorrectionNVSolver(
        geometry=geometry,
        parameters=fluid_params,
        sf_bcs=sf_bcs,
        sf_max_iters=geometry.n_y * geometry.n_x,
        sf_tolerance=1e-6,
        convective_term_form=ConvectiveTermForm.UPWIND,
    )

    start_time = time.perf_counter()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt

        u = heat_transfer_solver.solve(u=u, sf=sf, time=t)
        sf, w = navier_solver.solve(w=w, sf=sf, u=u, time=t)

        if n % 1000 == 0:
            t_min = t / 60
            if t_min % 4 == 0:
                np.savez_compressed(f"../data/icicle/4c/u_{n}.npz", u=u)
                print("bruh")
            dim_u = u * thermal_params.delta_u + thermal_params.u_ref
            delta = get_mushy_zone_temperature_range(
                u * thermal_params.delta_u + thermal_params.u_ref,
                u_pt=thermal_params.u_pt,
                h_x=geometry.dx,
                h_y=geometry.dy,
            )
            # plot_latent_heat_field(
            #     u=dim_u,
            #     u_pt=thermal_params.u_pt,
            #     delta=delta,
            #     l_solid=thermal_params.specific_latent_heat,
            #     geometry=geometry,
            #     graph_id=n,
            #     directory="../graphs/latent_heat/local_delta_take_2/",
            # )
            plot_temperature(
                u=dim_u,
                u_pt=thermal_params.u_pt,
                geometry=geometry,
                time=t,
                graph_id=n,
                plot_boundary=True,
                show_graph=False,
                min_temp=min_temp + ABS_ZERO,
                max_temp=max_temp + ABS_ZERO,
                actual_temp_units=TemperatureUnit.KELVIN,
                display_temp_units=TemperatureUnit.CELSIUS,
            )
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
            # j, i = np.unravel_index(sf.argmax(), sf.shape)
            # y, x = j * geometry.dy, i * geometry.dx
            # print(
            #     f"Maximum stream function value: {np.max(sf) * fluid_params.v * geometry.length_scale}, (x, y) = {x/geometry.length_scale:.3f}, {1.0 - y/geometry.length_scale:.3f}"
            # )
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

    print("Creating animation...")
    create_gif_from_images(output_filename="exp5", duration=200)
