#!/usr/bin/env python
"""
Generate the "warm start" precursor field for the freezing benchmark.

Following Kowalewski & Rebow (1997), the warm start begins from the steady state
established with the walls held at 10 degC and 0 degC; the right wall is only then
dropped to -10 degC. That steady state is resolution-dependent, so for a grid
refinement study it has to be recomputed on each grid rather than interpolated.

The material properties and the penalty/smoothing parameters are taken from the
freezing config so that the precursor and the production run use an identical
physical model — only the cold-wall temperature differs.

Examples
--------
    python -m src.examples.water_freezing.prepare_warm_start --nx 101 --ny 101
    python -m src.examples.water_freezing.prepare_warm_start --nx 201 --ny 201 --dt 0.25

Output goes to data/precursors/steady_<nx>x<ny>.npz, which is exactly where
run.py --start warm looks for it.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import BoundaryConditions
from src.core.constants import ABS_ZERO
from src.core.runner import ExperimentRunner, SimulationState
from src.fluid_dynamics.init_values import (
    initialize_stream_function,
    initialize_vorticity,
    initialize_velocity,
)
from src.fluid_dynamics.solvers import VorticitySolverName, StreamFunctionSolverName
from src.fluid_dynamics.solvers.bc_correction_solver_factory import BCCorrectionNVSolver
from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import PenaltyTermForm
from src.heat_transfer.coefficient_smoothing.coefficients import DeltaScheme, StepScheme
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.parameters.config import ExperimentConfig
from src.utils.boundary_conditions import (
    const_neumann_condition,
    const_dirichlet_condition,
)

HERE = Path(__file__).resolve().parent

T_HOT = 283.15  # +10 degC
T_COLD_PRECURSOR = 273.15  # 0 degC — the cold wall before the step change

logger = logging.getLogger("warm_start")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute the warm-start steady state for the freezing benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nx", type=int, default=None, help="grid points in x")
    p.add_argument("--ny", type=int, default=None, help="grid points in y")
    p.add_argument("--dt", type=float, default=0.5, help="time step [s]")
    p.add_argument(
        "--max-time",
        type=float,
        default=3600.0,
        help="upper bound on integration time [s]",
    )
    p.add_argument(
        "--steady-tol",
        type=float,
        default=1e-8,
        help="steady state when max|w^(n+1) - w^n| / dt falls below this",
    )
    p.add_argument(
        "--init-temp",
        type=float,
        default=5.0,
        help="uniform initial temperature [degC]",
    )
    p.add_argument(
        "--keep-phase-change",
        action="store_true",
        help="keep latent heat and the penalty term active. By default both are "
        "switched off: the cold wall of the precursor sits exactly at T_m, where the "
        "regularised delta function peaks and the liquid fraction is 0.5, so an active "
        "phase-change model would freeze and brake the wall layer in a state that the "
        "benchmark does not intend",
    )
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--out", type=Path, default=None, help="output npz path")
    p.add_argument("--log-interval", type=float, default=300.0, help="log every N s")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    with open(args.config, "r", encoding="utf-8") as f:
        data = json.load(f)

    geom = data["geometry"]
    if args.nx is not None:
        geom["n_x"] = args.nx
    if args.ny is not None:
        geom["n_y"] = args.ny

    n_t = int(round(args.max_time / args.dt))
    geom["end_time"] = n_t * args.dt
    geom["n_t"] = n_t

    if not args.keep_phase_change:
        data["material_props"]["specific_latent_heat"] = 1e-16

    return ExperimentConfig.model_validate(data)


class SteadyStateMonitor:
    """max|w^(n+1) - w^n| / dt, evaluated between successive steps."""

    def __init__(self, tol: float, dt: float):
        self.tol = tol
        self.dt = dt
        self._prev: np.ndarray | None = None
        self.last_residual = float("inf")

    def __call__(self, state: SimulationState) -> bool:
        if self._prev is None:
            self._prev = state.w.copy()
            return False
        self.last_residual = float(np.max(np.abs(state.w - self._prev)) / self.dt)
        self._prev[:] = state.w
        return self.last_residual < self.tol


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = build_config(args)
    geometry = cfg.geometry
    n_x, n_y, n_t, dt = geometry.n_x, geometry.n_y, geometry.n_t, geometry.dt
    delta_u, u_ref = cfg.delta_u, cfg.u_ref

    out = args.out or (HERE / "data" / "precursors" / f"steady_{n_x}x{n_y}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Warm-start precursor: grid=%dx%d dt=%g s, walls %.1f / %.1f degC, "
        "steady tol=%g, max %g s",
        n_x,
        n_y,
        dt,
        T_HOT + ABS_ZERO,
        T_COLD_PRECURSOR + ABS_ZERO,
        args.steady_tol,
        geometry.end_time,
    )

    u_bcs = BoundaryConditions(
        top=const_neumann_condition(n_x, value=0.0),
        right=const_dirichlet_condition(
            n_y, value=(T_COLD_PRECURSOR - u_ref) / delta_u
        ),
        bottom=const_neumann_condition(n_x, value=0.0),
        left=const_dirichlet_condition(n_y, value=(T_HOT - u_ref) / delta_u),
    )
    sf_bcs = BoundaryConditions(
        top=const_dirichlet_condition(n_x, value=0.0),
        right=const_dirichlet_condition(n_y, value=0.0),
        bottom=const_dirichlet_condition(n_x, value=0.0),
        left=const_dirichlet_condition(n_y, value=0.0),
    )

    u = init_temperature(
        cfg=cfg,
        bcs=u_bcs,
        shape=DomainShape.LINEAR,
        solid_temp=T_COLD_PRECURSOR,
        liquid_temp=T_HOT,
    )
    sf = initialize_stream_function(geometry=geometry, bcs=sf_bcs)
    w = initialize_vorticity(geometry=geometry)
    v_x, v_y = initialize_velocity(geometry=geometry)
    state = SimulationState(u=u, sf=sf, w=w, v_x=v_x, v_y=v_y)

    heat_solver = HeatTransferSolver(
        cfg=cfg,
        bcs=u_bcs,
        max_iters=1,
        tolerance=1e-6,
        urf=1.0,
        solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form=ConvectiveTermForm.DIVERGENT_CENTRAL,
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
        vorticity_bc_order=1,
    )
    if not args.keep_phase_change:
        navier_solver.vorticity_solver._calculate_penalty_term_coeff = (
            lambda *a, **kw: None
        )

    monitor = SteadyStateMonitor(tol=args.steady_tol, dt=dt)
    stride = max(1, int(round(args.log_interval / dt)))

    runner = ExperimentRunner(
        cfg=cfg,
        state=state,
        heat_solver=heat_solver,
        navier_solver=navier_solver,
        logger=logger,
        checkpoints_dir=out.parent,
        calculate_velocity=True,
        save_final=False,
        log_at=set(range(stride, n_t + 1, stride)),
        metrics={"steady_residual": lambda s: monitor.last_residual},
        stop_criterion=monitor,
    )

    t0 = time.perf_counter()
    runner.run()
    wall = time.perf_counter() - t0

    converged = monitor.last_residual < args.steady_tol
    if not converged:
        logger.warning(
            "Steady state NOT reached: residual %.3e > tol %.3e after %g s. "
            "Increase --max-time.",
            monitor.last_residual,
            args.steady_tol,
            state.t,
        )

    np.savez_compressed(
        out,
        n=state.n,
        t=state.t,
        u=state.u,
        sf=state.sf,
        w=state.w,
        v_x=state.v_x,
        v_y=state.v_y,
        steady_residual=monitor.last_residual,
        converged=converged,
        n_x=n_x,
        n_y=n_y,
        dt=dt,
    )
    logger.info(
        "Saved %s (t=%.1f s, residual=%.3e, converged=%s) in %.1f s wall clock",
        out,
        state.t,
        monitor.last_residual,
        converged,
        wall,
    )


if __name__ == "__main__":
    main()
