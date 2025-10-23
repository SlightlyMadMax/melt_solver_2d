import logging
import sys

from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import BoundaryConditions
from src.core.geometry import DomainGeometry
from src.core.runner import SimulationState, ExperimentRunner
from src.fluid_dynamics.init_values import (
    initialize_stream_function,
    initialize_vorticity,
    initialize_velocity,
)
from src.fluid_dynamics.solvers.bc_correction_solver_factory import BCCorrectionNVSolver
from src.heat_transfer.coefficient_smoothing.coefficients import StepScheme, DeltaScheme
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.plotting import plot_temperature
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    cfg: ExperimentConfig = ExperimentConfig.load_from_file(
        "../parameter_sets/gallium/config.json"
    )
    logger.info(cfg)
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
        graph_id=0,
        plot_boundary=True,
        show_graph=True,
    )

    heat_solver = HeatTransferSolver(
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

    logger.info((min_temp - u_ref) / delta_u)
    state = SimulationState(u=u, sf=sf, w=w, v_x=v_x, v_y=v_y)
    log_and_plot_interval = 60
    log_and_plot_at = set(
        [n for n in range(1, n_t + 1) if n * dt % log_and_plot_interval == 0]
    )
    runner = ExperimentRunner(
        cfg=cfg,
        state=state,
        heat_solver=heat_solver,
        navier_solver=navier_solver,
        checkpoints_dir="../data/gallium/test",
        logger=logger,
        save_at={
            int(2.0 * 60 / dt),
            int(3.0 * 60 / dt),
            int(6.0 * 60 / dt),
            int(8.0 * 60 / dt),
            int(10.0 * 60 / dt),
            int(12.5 * 60 / dt),
            int(15 * 60 / dt),
            int(17 * 60 / dt),
            int(19 * 60 / dt),
        },
        log_at=log_and_plot_at,
        plot_at=log_and_plot_at,
    )
    runner.run()
