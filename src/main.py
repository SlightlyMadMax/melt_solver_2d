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
    initialize_velocity,
)
from src.heat_transfer.coefficient_smoothing.coefficients import (
    StepScheme,
    DeltaScheme,
)
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.utils import TemperatureUnit
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
)
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.time_utils import get_remaining_time


if __name__ == "__main__":
    cfg = ExperimentConfig.load_from_file("../parameter_sets/octodecane/config.json")
    print(cfg)

    geometry: DomainGeometry = cfg.geometry

    dx, dy = geometry.dx, geometry.dy
    dt = geometry.dt
    n_x, n_y, n_t = geometry.n_x, geometry.n_y, geometry.n_t
    min_temp = 296.96
    max_temp = 305.7

    material_props: MaterialProperties = cfg.material_props

    delta_u = cfg.delta_u
    u_ref = cfg.u_ref
    u_pt = material_props.u_pt
    l = cfg.l
    v = cfg.v

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
        ),
    )

    # Initial temperature distribution
    u = init_temperature(
        cfg=cfg,
        bcs=u_bcs,
        shape=DomainShape.UNIFORM_SOLID,
        solid_temp=min_temp,
        liquid_temp=max_temp,
    )

    dim_u = u * delta_u + u_ref
    init_delta = get_mushy_zone_temperature_range(u=dim_u, u_pt=u_pt)

    print(f"Initial mushy zone temperature range: {init_delta:.2f}")

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

    # Initial stream function, vorticity and velocity fields
    sf = initialize_stream_function(geom=geometry, bcs=sf_bcs)
    w = initialize_vorticity(geom=geometry)
    v_x, v_y = initialize_velocity(geom=geometry)

    heat_transfer_solver = HeatTransferSolver(
        cfg=cfg,
        bcs=u_bcs,
        fixed_delta=False,
        max_iters=1,
        tolerance=1e-6,
        urf=1.0,
        solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form=ConvectiveTermForm.DIVERGENT_CENTRAL,
        bc_order=1,
        step_scheme=StepScheme.ERF,
        delta_scheme=DeltaScheme.GAUSS,
    )

    navier_solver = BCCorrectionNVSolver(
        cfg=cfg,
        sf_bcs=sf_bcs,
        sf_max_iters=(n_y - 2) * (n_x - 2),
        sf_tolerance=1e-6,
        convective_term_form=ConvectiveTermForm.DIVERGENT_CENTRAL,
    )

    start_time = time.perf_counter()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt
        u = heat_transfer_solver.solve(u=u, sf=sf, delta=0.01, time=t)
        sf, w = navier_solver.solve(w=w, sf=sf, u=u, delta=0.01, time=t)
        if n % cfg.save_interval == 0:
            u_dim = u * delta_u + u_ref
            sf_dim = sf * v * l
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
            )
            plot_stream_function(
                stream_function=sf_dim,
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

    # print("Creating animation...")
    # create_gif_from_images(output_filename="exp5", duration=200)
