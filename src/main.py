import time
import numpy as np

from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import BoundaryConditions
from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.init_values import (
    initialize_stream_function,
    initialize_vorticity,
    initialize_velocity,
)
from src.fluid_dynamics.plotting import plot_stream_function
from src.fluid_dynamics.solvers.bc_correction_solver_factory import BCCorrectionNVSolver
from src.heat_transfer.coefficient_smoothing.coefficients import StepScheme, DeltaScheme
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.utils import TemperatureUnit
from src.heat_transfer.plotting import plot_temperature
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)
from src.utils.time_utils import get_remaining_time


if __name__ == "__main__":
    cfg: ExperimentConfig = ExperimentConfig.load_from_file(
        "../parameter_sets/gallium/config.json"
    )
    print(cfg)
    geometry: DomainGeometry = cfg.geometry
    dx, dy, dt = geometry.dx, geometry.dy, geometry.dt
    n_x, n_y, n_t = geometry.n_x, geometry.n_y, geometry.n_t
    min_temp = 301.45
    max_temp = 311.15

    material_props: MaterialProperties = cfg.material_props

    delta_u = cfg.delta_u
    u_ref = cfg.u_ref
    u_pt = material_props.u_pt
    l = cfg.l
    v = cfg.v
    dt_scaled = dt * v / l

    # Temperature boundary conditions
    u_bcs = BoundaryConditions(
        top=const_neumann_condition(n_x, value=0.0),
        right=const_dirichlet_condition(n_y, value=(min_temp - u_ref) / delta_u),
        bottom=const_neumann_condition(n_x, value=0.0),
        left=const_dirichlet_condition(n_y, value=(max_temp - u_ref) / delta_u),
    )

    # Stream function boundary conditions
    sf_bcs = BoundaryConditions(
        top=const_dirichlet_condition(n_x, value=0.0),
        right=const_dirichlet_condition(n_y, value=0.0),
        bottom=const_dirichlet_condition(n_x, value=0.0),
        left=const_dirichlet_condition(n_y, value=0.0),
    )

    # Initial temperature distribution
    u = init_temperature(
        cfg=cfg,
        bcs=u_bcs,
        shape=DomainShape.UNIFORM_SOLID,
        solid_temp=min_temp,
        liquid_temp=max_temp,
    )

    # Initial stream function, vorticity and velocity fields
    sf = initialize_stream_function(geom=geometry, bcs=sf_bcs)
    w = initialize_vorticity(geom=geometry)
    v_x, v_y = initialize_velocity(geom=geometry)

    dim_u = u * delta_u + u_ref
    plot_temperature(
        u=dim_u,
        cfg=cfg,
        time=0.0,
        graph_id=0,
        plot_boundary=True,
        show_graph=True,
        min_temp=min_temp + ABS_ZERO,
        max_temp=max_temp + ABS_ZERO,
        actual_temp_units=TemperatureUnit.KELVIN,
        display_temp_units=TemperatureUnit.CELSIUS,
        # x_min=0.2,
        # x_max=0.4,
        # y_min=0.0,
        # y_max=0.2,
    )

    heat_transfer_solver = HeatTransferSolver(
        cfg=cfg,
        bcs=u_bcs,
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
        cfg=cfg,
        sf_bcs=sf_bcs,
        sf_max_iters=(n_y - 2) * (n_x - 2),
        sf_tolerance=1e-6,
        convective_term_form=ConvectiveTermForm.UPWIND,
        vorticity_bc_order=2,
    )

    print((min_temp - u_ref) / delta_u)
    delta = 0.008, 0.008
    start_time = time.perf_counter()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt
        u[:, :] = heat_transfer_solver.solve(u=u, sf=sf, delta=delta, time=t)
        sf[:, :], w[:, :] = navier_solver.solve(w=w, sf=sf, u=u, delta=0.008, time=t)

        t_min = t / 60
        if t_min in {2.0, 3.0, 6.0, 8.0, 10.0, 12.5, 15, 17, 19}:
            print("bruh")
            np.savez_compressed(f"../data/gallium/test/u_{n}.npz", u=u)

        # if t == 800.0 or t == 1575:
        #     print("bruh")
        #     np.savez_compressed(f"../data/octodecane/test/u_{n}.npz", u=u)
        # if t == 2340:
        #     print("bruh")
        #     np.savez_compressed(
        #         f"../data/water_freezing/after_freezing_201x201.npz", u=u, sf=sf, w=w
        #     )

        if n % cfg.save_interval == 0:
            u_dim = u * delta_u + u_ref
            sf_dim = sf * v * l
            # calculate_velocity_from_sf(sf_dim, v_x, v_y, dx, dy)

            plot_temperature(
                u=u_dim,
                cfg=cfg,
                time=t,
                graph_id=n,
                plot_boundary=True,
                show_graph=False,
                min_temp=min_temp + ABS_ZERO,
                max_temp=max_temp + ABS_ZERO,
                actual_temp_units=TemperatureUnit.KELVIN,
                display_temp_units=TemperatureUnit.CELSIUS,
                # x_min=0.2,
                # x_max=0.4,
                # y_min=0.0,
                # y_max=0.2,
            )
            # plot_stream_function(
            #     stream_function=sf_dim,
            #     geometry=geometry,
            #     graph_id=n,
            #     show_graph=False,
            # )
            print(
                f"Modelling Time: {n * dt:.2f} s, "
                f"Elapsed Time: {(time.perf_counter() - start_time) / 60:.2f} min., "
                f"Estimated Remaining Time: {get_remaining_time(n=n, n_t=n_t, start_time=start_time) / 60:.2f} min."
            )
            print(f"Maximum temperature value: {np.max(u_dim + ABS_ZERO):.2f} C")
            print(f"Minimum temperature value: {np.min(u_dim + ABS_ZERO):.2f} C")
            # j, i = np.unravel_index(sf_dim.argmax(), sf_dim.shape)
            # y, x = j * dy, i * dx
            # print(
            #     f"Maximum abs. stream function value: {np.max(np.abs(sf_dim)):.2e}, (x, y) = {x:.3f}, {1.0 - y:.3f}"
            # )
            # print(f"Minimum abs. stream function value: {np.min(np.abs(sf_dim)):.2e}")
            # print(
            #     f"Maximum vorticity value: {np.max(w) * v / l:.3f}"
            # )
            # print(
            #     f"Minimum vorticity value: {np.min(w) * v / l:.3f}"
            # )
            # print(f"Maximum speed: {np.max(np.sqrt(v_x**2 + v_y**2)):.2e}")
            # print(
            #     f"Maximum speed in the solid phase: {max_speed:.2e} at ({speed_ind[0] * dy:.3f}, {speed_ind[1] * dx:.3f})."
            # )
            # print(
            #     f"Maximum abs. stream function value in the solid phase: {max_sf:.2e} at ({sf_ind[0] * dy:.3f}, {sf_ind[1] * dx:.3f})."
            # )
            print()
