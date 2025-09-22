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
from src.fluid_dynamics.utils import calculate_velocity_from_sf
from src.heat_transfer.coefficient_smoothing.coefficients import (
    StepScheme,
    DeltaScheme,
)
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.heat_transfer.utils import TemperatureUnit
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
)
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.convection_benchmark import (
    calculate_U_profile_X05,
    calculate_T_profile_X05,
    calculate_W_profile_X05,
)
from src.utils.stand_with_icicle import init_temperature_icicle
from src.utils.time_utils import get_remaining_time


if __name__ == "__main__":
    cfg: ExperimentConfig = ExperimentConfig.load_from_file(
        "../parameter_sets/octodecane/config.json"
    )
    print(cfg)

    geometry: DomainGeometry = cfg.geometry

    dx, dy, dt = geometry.dx, geometry.dy, geometry.dt
    n_x, n_y, n_t = geometry.n_x, geometry.n_y, geometry.n_t
    min_temp = 301.2426
    max_temp = 310.07

    material_props: MaterialProperties = cfg.material_props

    delta_u = cfg.delta_u
    u_ref = cfg.u_ref
    u_pt = material_props.u_pt
    l = cfg.l
    v = cfg.v
    dt_scaled = dt * v / l

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

    print(f"Initial mushy zone temperature range: {max(init_delta):.2f}")

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
        convective_term_form=ConvectiveTermForm.UPWIND,
        bc_order=1,
        step_scheme=StepScheme.CONST,
        delta_scheme=DeltaScheme.GAUSS_ASYM,
    )

    navier_solver = BCCorrectionNVSolver(
        cfg=cfg,
        sf_bcs=sf_bcs,
        sf_max_iters=(n_y - 2) * (n_x - 2),
        sf_tolerance=1e-6,
        convective_term_form=ConvectiveTermForm.UPWIND,
    )

    # delta = 0.01, 0.01
    start_time = time.perf_counter()
    for n in range(1, geometry.n_t):
        t = n * geometry.dt
        delta = get_mushy_zone_temperature_range(u=u, u_pt=cfg.u_pt_nd, n_nodes=2)
        u = heat_transfer_solver.solve(u=u, sf=sf, delta=delta, time=t)
        delta = get_mushy_zone_temperature_range(u=u, u_pt=cfg.u_pt_nd, n_nodes=2)
        sf, w = navier_solver.solve(w=w, sf=sf, u=u, delta=max(delta), time=t)

        if n % cfg.save_interval == 0:
            u_dim = u * delta_u + u_ref
            sf_dim = sf * v * l
            calculate_velocity_from_sf(sf_dim, v_x, v_y, dx, dy)

            # from matplotlib import pyplot as plt
            #
            # speed = np.sqrt(v_x**2 + v_y**2)
            #
            # X, Y = geometry.mesh_grid
            # plt.figure(figsize=(8, 6))
            # ax = plt.axes(
            #     xlim=(0, cfg.geometry.width),
            #     ylim=(0, cfg.geometry.height),
            #     xlabel="x, м",
            #     ylabel="y, м",
            # )
            # contour = plt.contourf(
            #     X,
            #     Y,
            #     speed,
            #     25,
            #     cmap="viridis",
            #     extend="both",
            # )
            # cbar = plt.colorbar(contour)
            #
            # X_b, Y_b = get_phase_trans_boundary(u=u_dim, cfg=cfg)
            # plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color="r", label="Граница ф.п.")
            # ax.legend()
            #
            # plt.show()
            #
            # from matplotlib import pyplot as plt
            #
            # diff = u - cfg.u_pt_nd
            # # norm_coeff = 2.0 / (np.sqrt(2 * np.pi) * (delta[0] + delta[1]))
            # # latent = np.where(
            # #     diff <= 0,
            # #     norm_coeff * np.exp(-(diff**2) / (2 * delta[0]**2)),
            # #     norm_coeff * np.exp(-(diff**2) / (2 * delta[1]**2)),
            # # )
            # latent = np.where(abs(diff) <= min(delta), 0.5 / min(delta), 0.0)
            # print(delta[0], delta[1])
            # X, Y = geometry.mesh_grid
            # plt.figure(figsize=(8, 6))
            # ax = plt.axes(
            #     xlim=(0, cfg.geometry.width),
            #     ylim=(0, cfg.geometry.height),
            #     xlabel="x, м",
            #     ylabel="y, м",
            # )
            # contour = plt.contourf(
            #     X,
            #     Y,
            #     latent,
            #     25,
            #     cmap="viridis",
            #     extend="both",
            # )
            # cbar = plt.colorbar(contour)
            #
            # X_b, Y_b = get_phase_trans_boundary(u=u_dim, cfg=cfg)
            # plt.scatter(X_b, Y_b, s=1, linewidths=0.1, color="r", label="Граница ф.п.")
            # ax.legend()
            #
            # plt.show()

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
