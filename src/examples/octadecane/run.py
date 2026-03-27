import logging
import sys

import numpy as np

from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import BoundaryConditions
from src.core.constants import ABS_ZERO
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
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.plotting import plot_temperature
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)
from src.utils.nusselt import calculate_nusselt

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def nusselt_correlation(tau, Ra, c1=0.27, c2=0.0275, n=-2):
    """
    Compute the average Nusselt number using the Jany & Bejan (1988) correlation.

    Nu(τ) = 1/√(2τ) + [c₁·Ra^(1/4) - 1/√(2τ)]·[1 + (c₂·Ra^(3/4)·τ^(3/2))^n]^(1/n)

    Parameters
    ----------
    tau : float or np.ndarray
        Dimensionless time τ = Fo·Ste (Fourier·Stefan)
    Ra : float
        Rayleigh number
    c1 : float, optional (default 0.27)
        Constant c₁ from the correlation
    c2 : float, optional (default 0.0275)
        Constant c₂ from the correlation
    n : float, optional (default -2)
        Exponent n from the correlation

    Returns
    -------
    Nu : float or np.ndarray
        Average Nusselt number at the hot wall
    """
    # Convert to array if not scalar
    tau = np.asarray(tau) if not np.isscalar(tau) else tau

    # Check for non-positive tau
    if np.any(tau <= 0):
        raise ValueError("tau must be positive")

    # Pure conduction term (Neumann solution)
    nu_conduction = 1.0 / np.sqrt(2.0 * tau)

    # Pure convection limit
    nu_convection = c1 * np.power(Ra, 1.0 / 4.0)

    # Correlation factor for transition regime
    correlation_factor = np.power(
        1.0 + np.power(c2 * np.power(Ra, 3.0 / 4.0) * np.power(tau, 3.0 / 2.0), n),
        1.0 / n,
    )

    # Full formula
    Nu = nu_conduction + (nu_convection - nu_conduction) * correlation_factor

    return Nu


if __name__ == "__main__":
    cfg: ExperimentConfig = ExperimentConfig.load_from_file("./config.json")
    logger.info(cfg)
    geometry: DomainGeometry = cfg.geometry
    dt = geometry.dt
    n_x, n_y, n_t = geometry.n_x, geometry.n_y, geometry.n_t
    min_temp = 301.2426
    max_temp = 310.07

    material_props: MaterialProperties = cfg.material_props

    delta_u = cfg.delta_u
    u_ref = cfg.u_ref

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

    dim_u = u * delta_u + u_ref
    plot_temperature(
        u=dim_u,
        cfg=cfg,
        graph_id=0,
        plot_boundary=True,
        show_graph=True,
    )

    # Initial stream function, vorticity and velocity fields
    sf = initialize_stream_function(geometry=geometry, bcs=sf_bcs)
    w = initialize_vorticity(geometry=geometry)
    v_x, v_y = initialize_velocity(geometry=geometry)

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
    )

    navier_solver = BCCorrectionNVSolver(
        cfg=cfg,
        sf_bcs=sf_bcs,
        sf_max_iters=(n_y - 2) * (n_x - 2),
        sf_tolerance=1e-6,
        convective_term_form=ConvectiveTermForm.DIVERGENT_CENTRAL,
        penalty_term_form=PenaltyTermForm.QUADRATIC,
        vorticity_solver_name=VorticitySolverName.PEACEMAN_RACHFORD,
        stream_function_solver_name=StreamFunctionSolverName.AMG,
        vorticity_bc_order=2,
    )

    state = SimulationState(u=u, sf=sf, w=w, v_x=v_x, v_y=v_y)

    log_interval = 60
    plot_interval = 60
    log_at = set([n for n in range(1, n_t + 1) if n * dt % log_interval == 0])
    plot_at = set([n for n in range(1, n_t + 1) if n * dt % plot_interval == 0])
    save_at = {int(800 / dt), int(1575 / dt)}

    nu_history = []

    runner = ExperimentRunner(
        cfg=cfg,
        state=state,
        heat_solver=heat_solver,
        navier_solver=navier_solver,
        logger=logger,
        checkpoints_dir=f"./data",
        calculate_velocity=False,
        plot_at=plot_at,
        log_at=log_at,
        save_at=save_at,
        metrics={
            "T_max, °C": lambda s: np.max(s.u * cfg.delta_u + cfg.u_ref + ABS_ZERO),
            "T_min, °C": lambda s: np.min(s.u * cfg.delta_u + cfg.u_ref + ABS_ZERO),
        },
        step_callback=lambda s: nu_history.append(
            (s.t, calculate_nusselt(s.u, cfg, order=2))
        ),
    )
    runner.run()

    import matplotlib.pyplot as plt
    import numpy as np

    # Распаковка данных
    times, nu_values = zip(*nu_history)

    dim_times = [
        t * cfg.stefan_number * cfg.thermal_diffusivity_ref / cfg.l**2 for t in times
    ]
    nu_pred = nusselt_correlation(dim_times, Ra=cfg.rayleigh_number)

    plt.figure(figsize=(8, 5))
    plt.plot(dim_times[1000:], nu_values[1000:], linewidth=1.5, label="Nu(t)")
    plt.plot(dim_times[1000:], nu_pred[1000:], linewidth=1.5, label="Prediction")

    plt.xlabel(r"Безразмерное время $\tau = Fo \cdot Ste$")
    plt.ylabel(r"Среднее число Нуссельта $\overline{Nu}$")
    plt.title("Эволюция теплопередачи на горячей стенке")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./graphs/nusselt_evolution_3.png", dpi=300)
