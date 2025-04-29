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
    IterativeNavierStokesSolver,
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
from src.utils import get_remaining_time


if __name__ == "__main__":
    geometry = DomainGeometry(
        width=0.4,
        height=0.4,
        end_time=60.0 * 60.0 * 24.0,
        n_x=401,
        n_y=401,
        n_t=60 * 60 * 24 * 200,
    )

    print(geometry)

    min_temp = 273.14
    max_temp = 281.15
    reference_temperature = 0.5 * (min_temp + max_temp)
    delta_u = (max_temp - min_temp) / 2
    # reference_temperature = max_temp

    thermal_params = ThermalParameters(
        u_pt=273.15,
        u_ref=reference_temperature,
        delta_u=delta_u,
        v=0.04,
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

    fluid_params = FluidParameters(
        u_pt=273.15,
        u_ref=reference_temperature,
        delta_u=delta_u,
        v=0.04,
        l=geometry.length_scale,
        epsilon=0.00075,
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

    u = init_temperature(
        geom=geometry,
        thermal_parameters=thermal_params,
        shape=DomainShape.RECTANGLE,
        rect_width=0.04,
        rect_height=0.12,
        solid_temp=max_temp,
        liquid_temp=min_temp,
    )

    # u[:, 0] = (max_temp - thermal_params.u_ref) / thermal_params.delta_u
    # u[0:3, :] = (min_temp - thermal_params.u_ref) / thermal_params.delta_u

    init_delta = get_max_delta(
        u * thermal_params.delta_u + thermal_params.u_ref, u_pt=thermal_params.u_pt
    )
    print(f"Delta for the initial temperature distribution: {init_delta:.2f}")

    # plot_temperature(
    #     u=u * thermal_params.delta_u + thermal_params.u_ref,
    #     u_pt=thermal_params.u_pt,
    #     geom=geometry,
    #     time=0.0,
    #     graph_id=0,
    #     plot_boundary=True,
    #     show_graph=True,
    #     min_temp=min_temp + ABS_ZERO,
    #     max_temp=max_temp + ABS_ZERO,
    #     actual_temp_units=TemperatureUnit.KELVIN,
    #     display_temp_units=TemperatureUnit.CELSIUS,
    # )

    # Temperature boundary conditions
    u_bcs = BoundaryConditions(
        top=BoundaryCondition(
            boundary_type=BoundaryConditionType.NEUMANN,
            n=geometry.n_x,
            flux_func=lambda t, n: np.zeros(n),
            # value_func=lambda t, n: (min_temp - thermal_params.u_ref)
            # / thermal_params.delta_u
            # * np.ones(n),
        ),
        right=BoundaryCondition(
            boundary_type=BoundaryConditionType.NEUMANN,
            n=geometry.n_y,
            flux_func=lambda t, n: np.zeros(n),
            # value_func=lambda t, n: (min_temp - thermal_params.u_ref)
            # / thermal_params.delta_u
            # * np.ones(n),
        ),
        bottom=BoundaryCondition(
            boundary_type=BoundaryConditionType.NEUMANN,
            n=geometry.n_x,
            flux_func=lambda t, n: np.zeros(n),
            # value_func=lambda t, n: (272.15 - thermal_params.u_ref)
            # / thermal_params.delta_u
            # * np.ones(n),
        ),
        left=BoundaryCondition(
            boundary_type=BoundaryConditionType.NEUMANN,
            n=geometry.n_y,
            flux_func=lambda t, n: np.zeros(n),
            # value_func=lambda t, n: (max_temp - thermal_params.u_ref)
            # / thermal_params.delta_u
            # * np.ones(n),
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
        solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form=ConvectiveTermForm.UPWIND,
    )

    navier_solver = IterativeNavierStokesSolver(
        geometry=geometry,
        parameters=fluid_params,
        sf_bcs=sf_bcs,
        vorticity_solver_name=VorticitySolverName.PEACEMAN_RACHFORD,
        convective_term_form=ConvectiveTermForm.UPWIND,
        stream_function_solver_name=StreamFunctionSolverName.MATRIX_SWEEP,
        implicit_lin_max_iters=1,
        implicit_lin_urf=1.0,
    )

    # navier_solver = NonIterativeNavierStokersSolver(
    #     geometry=geometry,
    #     parameters=fluid_params,
    #     sf_bcs=sf_bcs,
    #     sf_max_iters=geometry.n_x * geometry.n_y,
    #     sf_stopping_criteria=1e-5,
    # )

    start_time = time.perf_counter()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt

        u = heat_transfer_solver.solve(u=u, sf=sf, time=t)
        sf, w = navier_solver.solve(w=w, sf=sf, u=u, time=t)

        if n % 2000 == 0:
            np.savez_compressed(f"../data/experiment/u_{n}.npz", u=u)
            d = get_max_delta(
                u * thermal_params.delta_u + thermal_params.u_ref,
                u_pt=thermal_params.u_pt,
            )
            if d <= 0.0:
                break
            # plot_temperature(
            #     u=u * thermal_params.delta_u + thermal_params.u_ref,
            #     u_pt=thermal_params.u_pt,
            #     geom=geometry,
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
                f"Modelling Time: {n * geometry.dt} s, "
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
            #     f"Maximum stream function value: {np.max(sf) * fluid_params.v * geometry.length_scale / thermal_params.thermal_diffusivity_solid:.3f}, (x, y) = {x:.3f}, {1.0 - y:.3f}"
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

    # print("Creating animation...")
    # create_gif_from_images(output_filename="hyper_coeff.gif", duration=300)
