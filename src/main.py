import logging
import sys

import numpy as np

from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.core.runner import SimulationState, ExperimentRunner
from src.fluid_dynamics.init_values import (
    initialize_stream_function,
    initialize_vorticity,
    initialize_velocity,
)
from src.fluid_dynamics.solvers import VorticitySolverName, StreamFunctionSolverName
from src.fluid_dynamics.solvers.bc_correction_solver_factory import BCCorrectionNVSolver
from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import PenaltyTermForm
from src.heat_transfer.coefficient_smoothing.coefficients import StepScheme, DeltaScheme
from src.heat_transfer.init_values import (
    init_temperature,
    DomainShape,
    init_temperature_with_interface,
)
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)
from src.utils.icicle import init_temperature_icicle

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    cfg: ExperimentConfig = ExperimentConfig.load_from_file(
        "../parameter_sets/water/horizontal_layer.json"
    )
    logger.info(cfg)
    geometry: DomainGeometry = cfg.geometry
    dt = geometry.dt
    n_x, n_y, n_t = geometry.n_x, geometry.n_y, geometry.n_t
    min_temp = 268.15
    max_temp = 278.15

    material_props: MaterialProperties = cfg.material_props

    delta_u = cfg.delta_u
    u_ref = cfg.u_ref

    # Temperature boundary conditions
    u_bcs = BoundaryConditions(
        # top=const_neumann_condition(n_x, value=0.0),
        top=const_dirichlet_condition(n_x, value=(max_temp - u_ref) / delta_u),
        # right=const_dirichlet_condition(n_y, value=(min_temp - u_ref) / delta_u),
        right=const_neumann_condition(n_y, value=0.0),
        # bottom=const_neumann_condition(n_x, value=0.0),
        bottom=const_dirichlet_condition(n_x, value=(min_temp - u_ref) / delta_u),
        # left=const_dirichlet_condition(n_y, value=(max_temp - u_ref) / delta_u),
        left=const_neumann_condition(n_y, value=0.0),
    )

    # Stream function boundary conditions
    sf_bcs = BoundaryConditions(
        top=const_dirichlet_condition(n_x, value=0.0),
        right=const_dirichlet_condition(n_y, value=0.0),
        bottom=const_dirichlet_condition(n_x, value=0.0),
        left=const_dirichlet_condition(n_y, value=0.0),
    )

    # Initial temperature distribution
    # u = init_temperature(
    #     cfg=cfg,
    #     bcs=u_bcs,
    #     shape=DomainShape.LINEAR,
    #     solid_temp=min_temp,
    #     liquid_temp=max_temp,
    # )
    # u = init_temperature_icicle(
    #     cfg=cfg,
    #     liquid_temp=max_temp,
    #     solid_temp=min_temp,
    #     rect_width=0.09,
    #     rect_height=0.12,
    #     location="top",
    # )

    water_thickness = 0.025
    # crevasse_width = 0.02
    # crevasse_depth = 0.2
    f = np.empty(n_x)

    # angle_rad = np.deg2rad(15.0)
    # tan15 = np.tan(angle_rad)
    # w_half = crevasse_depth * tan15
    #
    # for i in range(n_x):
    #     x = i * geometry.dx
    #     dx = abs(x - geometry.width / 2)
    #
    #     if dx <= w_half:
    #         local_depth = crevasse_depth - dx / tan15
    #         f[i] = geometry.height - water_thickness - local_depth
    #     else:
    #         f[i] = geometry.height - water_thickness

    # for i in range(n_x):
    #     x = i * geometry.dx
    #     if abs(x - geometry.width / 2) <= crevasse_width / 2:
    #         f[i] = geometry.height - water_thickness - crevasse_depth
    #     else:
    #         f[i] = geometry.height - water_thickness

    for i in range(n_x):
        x = i * geometry.dx
        f[i] = geometry.height - water_thickness

    u = init_temperature_with_interface(
        cfg=cfg,
        f=f,
        liquid_region_height=water_thickness,
        liquid_temp=max_temp,
        solid_temp=min_temp,
    )

    # Initial stream function, vorticity and velocity fields
    sf = initialize_stream_function(geometry=geometry, bcs=sf_bcs)
    w = initialize_vorticity(geometry=geometry)
    v_x, v_y = initialize_velocity(geometry=geometry)

    # dim_u = u * delta_u + u_ref
    # plot_temperature(
    #     u=dim_u,
    #     cfg=cfg,
    #     graph_id=0,
    #     plot_boundary=True,
    #     show_graph=True,
    # )

    heat_solver = HeatTransferSolver(
        cfg=cfg,
        bcs=u_bcs,
        max_iters=1,
        tolerance=1e-6,
        urf=1.0,
        solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form=ConvectiveTermForm.DEFERRED_CORRECTION,
        step_scheme=StepScheme.ERF,
        delta_scheme=DeltaScheme.GAUSS,
        k_face_method=KFaceMethod.FROM_TEMP,
        post_correction=False,
    )

    navier_solver = BCCorrectionNVSolver(
        cfg=cfg,
        sf_bcs=sf_bcs,
        sf_max_iters=(n_y - 2) * (n_x - 2),
        sf_tolerance=1e-6,
        convective_term_form=ConvectiveTermForm.DIVERGENT_CENTRAL,
        penalty_term_form=PenaltyTermForm.TANH,
        vorticity_solver_name=VorticitySolverName.PEACEMAN_RACHFORD,
        stream_function_solver_name=StreamFunctionSolverName.AMG,
        vorticity_bc_order=2,
    )

    state = SimulationState(u=u, sf=sf, w=w, v_x=v_x, v_y=v_y)

    log_interval = 60
    plot_interval = 60
    save_interval = 60
    log_at = set([n for n in range(1, n_t + 1) if n * dt % log_interval == 0])
    plot_at = set([n for n in range(1, n_t + 1) if n * dt % plot_interval == 0])
    save_at = set([n for n in range(1, n_t + 1) if n * dt % save_interval == 0])

    # save_at = {
    #     int(2.0 * 60 / dt),
    #     int(3.0 * 60 / dt),
    #     int(6.0 * 60 / dt),
    #     int(8.0 * 60 / dt),
    #     int(10.0 * 60 / dt),
    #     int(12.5 * 60 / dt),
    #     int(15.0 * 60 / dt),
    #     int(17.0 * 60 / dt),
    #     int(19.0 * 60 / dt),
    # }
    # save_at = {int(800 / dt), int(1575 / dt)}

    runner = ExperimentRunner(
        cfg=cfg,
        state=state,
        heat_solver=heat_solver,
        navier_solver=navier_solver,
        checkpoints_dir="../data/wavy_surface/5x20_3",
        logger=logger,
        save_at=save_at,
        log_at=log_at,
        plot_at=set(),
        calculate_velocity=False,
    )
    # runner.register_callback(event="on_plot", fn=on_plot)
    runner.run()
