#!/usr/bin/env python
"""
Melting of n-octadecane in a square cavity (Okada 1984 benchmark).

A 15 mm square, initially solid at 28.09 degC, just below the melting point. The left
wall is raised to 36.92 degC and the melt front travels to the right; the right wall
is held at the initial temperature and the horizontal walls are adiabatic. The
published comparisons are the front at t = 800 s and t = 1575 s.

Only the parameters a sensitivity study actually varies are on the command line; the
material properties, the wall temperatures and the geometry stay in config.json.

Examples
--------
Reference run, as in the manuscript:
    python -m src.examples.octadecane.run

New scheme (implicit penalty, central divergent convection, no latent heat in the
convective term):
    python -m src.examples.octadecane.run --penalty-time-scheme implicit \
        --heat-convection central-div --no-latent-convection --tag newscheme

Grid and time-step study:
    python -m src.examples.octadecane.run --nx 201 --ny 201 --dt 0.05
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

from src.convective_operators import ConvectiveTermForm
from src.core.boundary_conditions import BoundaryConditions
from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.core.runner import ExperimentRunner, SimulationState
from src.fluid_dynamics.init_values import (
    initialize_stream_function,
    initialize_velocity,
    initialize_vorticity,
)
from src.fluid_dynamics.solvers import StreamFunctionSolverName, VorticitySolverName
from src.fluid_dynamics.solvers.bc_correction_solver_factory import BCCorrectionNVSolver
from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import PenaltyTermForm
from src.fluid_dynamics.utils import calculate_velocity_from_sf
from src.heat_transfer.coefficient_smoothing.coefficients import DeltaScheme, StepScheme
from src.heat_transfer.init_values import DomainShape, init_temperature
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.parameters.config import ExperimentConfig
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)
from src.utils.nusselt import calculate_nusselt

HERE = Path(__file__).resolve().parent

# Okada's cell: the hot wall is stepped up to T_h at t = 0, the cold wall stays at the
# initial temperature of the solid, which is 0.09 K below the melting point
T_HOT = 310.07  # K
T_COLD = 301.2426  # K

# The instants the published fronts are given at
COMPARISON_TIMES = (800.0, 1575.0)

logger = logging.getLogger("octadecane")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Melting of n-octadecane in a square cavity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grid = p.add_argument_group("grid and time")
    grid.add_argument("--nx", type=int, default=None, help="grid points in x")
    grid.add_argument("--ny", type=int, default=None, help="grid points in y")
    grid.add_argument("--dt", type=float, default=None, help="time step [s]")
    grid.add_argument(
        "--end-time", type=float, default=1600.0, help="final physical time [s]"
    )

    phys = p.add_argument_group("phase-change / penalty parameters")
    phys.add_argument(
        "--eps-t",
        type=float,
        default=None,
        help="half-width of the smoothing interval for the heat equation [K]",
    )
    phys.add_argument(
        "--eps-flow",
        type=float,
        default=None,
        help="half-width of the smoothing interval for the penalty term [K]; "
        "defaults to --eps-t when that is given",
    )
    phys.add_argument(
        "--penalty-c",
        type=float,
        default=None,
        help="penalty coefficient C [1/s]; stored in the config as epsilon = 1/sqrt(C)",
    )

    scheme = p.add_argument_group("scheme")
    scheme.add_argument(
        "--penalty-time-scheme",
        choices=("cn", "dr", "implicit"),
        default="cn",
        help="time discretisation of the penalty term; 'cn' is the published scheme",
    )
    scheme.add_argument(
        "--heat-convection",
        choices=("deferred", "central-div", "deferred-div"),
        default="deferred",
        help="convective term of the heat equation: first-order upwind with limited "
        "deferred correction, central differences in divergent form d(v u)/dx, or "
        "upwind with limited deferred correction written as face fluxes of d(v u)/dx",
    )
    scheme.add_argument(
        "--no-latent-convection",
        action="store_true",
        help="multiply the convective term of the heat equation by c instead of "
        "c + lambda*delta, so that the flow carries sensible heat only",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="output directory; auto-named under data/scratch/ when omitted",
    )
    out.add_argument("--tag", type=str, default=None, help="extra label for the outdir")
    out.add_argument(
        "--save-times",
        type=float,
        nargs="*",
        default=list(COMPARISON_TIMES),
        help="physical times to checkpoint at [s]; the final state is always saved",
    )
    out.add_argument(
        "--log-interval", type=float, default=60.0, help="log interval [s]"
    )
    out.add_argument("--quiet", action="store_true", help="only warnings and errors")

    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """
    Apply the CLI overrides to the JSON payload *before* validation.

    ExperimentConfig caches derived quantities via cached_property, so mutating an
    already-constructed instance would leave those stale.
    """
    with open(HERE / "config.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    geom = data["geometry"]
    if args.nx is not None:
        geom["n_x"] = args.nx
    if args.ny is not None:
        geom["n_y"] = args.ny

    dt = args.dt if args.dt is not None else geom["end_time"] / geom["n_t"]
    n_t = int(round(args.end_time / dt))
    if n_t < 1:
        raise SystemExit(f"end_time={args.end_time} is shorter than dt={dt}")
    # Keep dt exact: end_time is snapped to a whole number of steps
    geom["end_time"] = n_t * dt
    geom["n_t"] = n_t

    if args.eps_t is not None:
        data["delta"] = args.eps_t
    if args.eps_flow is not None:
        data["delta_flow"] = args.eps_flow
    elif args.eps_t is not None:
        data["delta_flow"] = args.eps_t

    if args.penalty_c is not None:
        # The solver builds the dimensionless penalty as l / (epsilon**2 * v),
        # which corresponds to a dimensional C = 1 / epsilon**2 [1/s]
        data["epsilon"] = 1.0 / math.sqrt(args.penalty_c)

    return ExperimentConfig.model_validate(data)


def penalty_c_of(cfg: ExperimentConfig) -> float:
    """Dimensional penalty coefficient C [1/s] implied by cfg.epsilon."""
    return 1.0 / cfg.epsilon**2


def auto_outdir(args: argparse.Namespace, cfg: ExperimentConfig) -> Path:
    if args.outdir is not None:
        return args.outdir
    g = cfg.geometry
    parts = [
        f"{g.n_x}x{g.n_y}",
        f"dt{g.dt:g}",
        f"epsT{cfg.delta:g}",
        f"epsF{cfg.delta_flow:g}",
        f"C{penalty_c_of(cfg):.0e}",
        args.penalty_time_scheme,
        args.heat_convection,
    ]
    if args.no_latent_convection:
        parts.append("nolatconv")
    if args.tag:
        parts.append(args.tag)
    return HERE / "data" / "scratch" / "_".join(parts)


def liquid_fraction(u: np.ndarray, cfg: ExperimentConfig) -> float:
    """Area fraction of the domain above the phase-change temperature."""
    return float(np.mean(u > cfg.u_pt_nd))


def interface_positions(u: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """
    Horizontal position of the T = T_m isotherm in every row [m], nan where a row holds
    a single phase.

    The melt grows from the hot (left) wall, so each row is scanned from the left for
    its first crossing, which is refined by linear interpolation.
    """
    u_pt = cfg.u_pt_nd
    dx = cfg.geometry.dx
    diff = u - u_pt
    crossing = diff[:, :-1] * diff[:, 1:] < 0  # (n_y, n_x - 1)
    positions = np.full(u.shape[0], np.nan)
    rows = np.nonzero(crossing.any(axis=1))[0]
    i = np.argmax(crossing, axis=1)[rows]
    a, b = u[rows, i], u[rows, i + 1]
    positions[rows] = (i + (u_pt - a) / (b - a)) * dx
    return positions


def mean_interface_x(u: np.ndarray, cfg: ExperimentConfig) -> float:
    x = interface_positions(u, cfg)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def speeds(state: SimulationState, cfg: ExperimentConfig) -> tuple[float, float]:
    """Largest speed in the liquid and in the solid [m/s], from the stream function."""
    v_x, v_y = np.empty_like(state.sf), np.empty_like(state.sf)
    calculate_velocity_from_sf(state.sf, v_x, v_y, cfg)
    speed = np.hypot(v_x, v_y) * cfg.v
    liquid = state.u > cfg.u_pt_nd
    in_liquid = float(speed[liquid].max()) if liquid.any() else 0.0
    in_solid = float(speed[~liquid].max()) if (~liquid).any() else 0.0
    return in_liquid, in_solid


def steps_at_times(times, dt: float, n_t: int) -> set[int]:
    """Step indices closest to the given physical times."""
    return {min(max(1, int(round(t / dt))), n_t) for t in times if t > 0.0}


def steps_at_interval(interval: float, dt: float, n_t: int) -> set[int]:
    """
    Step indices closest to multiples of `interval`.

    Integer arithmetic on purpose: `n * dt % interval == 0` silently misses events
    because n * dt is almost never an exact multiple of the interval in floating point.
    """
    if interval is None or interval <= 0.0:
        return set()
    stride = max(1, int(round(interval / dt)))
    return set(range(stride, n_t + 1, stride))


def run(args: argparse.Namespace) -> dict:
    cfg = build_config(args)
    geometry: DomainGeometry = cfg.geometry
    n_x, n_y, n_t, dt = geometry.n_x, geometry.n_y, geometry.n_t, geometry.dt

    outdir = auto_outdir(args, cfg)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(cfg)
    logger.info(
        "Run: grid=%dx%d dt=%g s end=%g s (%d steps) eps_T=%g K eps_flow=%g K "
        "C=%.3g 1/s penalty=%s convection=%s%s",
        n_x,
        n_y,
        dt,
        geometry.end_time,
        n_t,
        cfg.delta,
        cfg.delta_flow,
        penalty_c_of(cfg),
        args.penalty_time_scheme,
        args.heat_convection,
        ", no latent heat in convection" if args.no_latent_convection else "",
    )
    logger.info("Output: %s", outdir)

    delta_u, u_ref = cfg.delta_u, cfg.u_ref

    u_bcs = BoundaryConditions(
        top=const_neumann_condition(n_x, value=0.0),
        right=const_dirichlet_condition(n_y, value=(T_COLD - u_ref) / delta_u),
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
        shape=DomainShape.UNIFORM_SOLID,
        solid_temp=T_COLD,
        liquid_temp=T_HOT,
    )

    v_x, v_y = initialize_velocity(geometry=geometry)
    state = SimulationState(
        u=u,
        sf=initialize_stream_function(geometry=geometry, bcs=sf_bcs),
        w=initialize_vorticity(geometry=geometry),
        v_x=v_x,
        v_y=v_y,
    )

    heat_solver = HeatTransferSolver(
        cfg=cfg,
        bcs=u_bcs,
        max_iters=1,
        tolerance=1e-6,
        urf=1.0,
        solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
        convective_term_form={
            "deferred": ConvectiveTermForm.DEFERRED_CORRECTION,
            "central-div": ConvectiveTermForm.DIVERGENT_CENTRAL,
            "deferred-div": ConvectiveTermForm.DEFERRED_CORRECTION_DIV,
        }[args.heat_convection],
        step_scheme=StepScheme.ERF,
        delta_scheme=DeltaScheme.GAUSS,
        k_face_method=KFaceMethod.FROM_TEMP,
        latent_convection=not args.no_latent_convection,
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
        penalty_time_scheme=args.penalty_time_scheme,
    )

    nu_history = []
    save_at = steps_at_times(args.save_times, dt, n_t)

    runner = ExperimentRunner(
        cfg=cfg,
        state=state,
        heat_solver=heat_solver,
        navier_solver=navier_solver,
        logger=logger,
        checkpoints_dir=outdir,
        calculate_velocity=False,
        save_at=save_at,
        save_final=True,
        log_at=steps_at_interval(args.log_interval, dt, n_t),
        metrics={
            "liquid_fraction": lambda s: liquid_fraction(s.u, cfg),
            "mean_interface_x, mm": lambda s: 1e3 * mean_interface_x(s.u, cfg),
            "Nu_hot": lambda s: calculate_nusselt(s.u, cfg, wall="left", order=2),
            "T_max, °C": lambda s: np.max(s.u * delta_u + u_ref + ABS_ZERO),
        },
        step_callback=lambda s: nu_history.append(
            (s.t, calculate_nusselt(s.u, cfg, wall="left", order=2))
        ),
    )

    wall_t0 = time.perf_counter()
    runner.run()
    wall = time.perf_counter() - wall_t0

    np.savez_compressed(outdir / "nusselt.npz", nu=np.asarray(nu_history))

    v_liquid, v_solid = speeds(state, cfg)
    summary = {
        "n_x": n_x,
        "n_y": n_y,
        "dx": geometry.dx,
        "dy": geometry.dy,
        "dt": dt,
        "end_time": geometry.end_time,
        "n_t": n_t,
        "t_hot": T_HOT,
        "t_cold": T_COLD,
        "eps_T": cfg.delta,
        "eps_flow": cfg.delta_flow,
        "penalty_C": penalty_c_of(cfg),
        "penalty_time_scheme": args.penalty_time_scheme,
        "heat_convection": args.heat_convection,
        "latent_convection": not args.no_latent_convection,
        "Ra": cfg.rayleigh_number,
        "Pr": cfg.prandtl_number,
        "Ste": cfg.stefan_number,
        "liquid_fraction": liquid_fraction(state.u, cfg),
        "mean_interface_x": mean_interface_x(state.u, cfg),
        "Nu_hot": calculate_nusselt(state.u, cfg, wall="left", order=2),
        "Nu_cold": calculate_nusselt(state.u, cfg, wall="right", order=2),
        "max_speed_liquid": v_liquid,
        "max_speed_solid": v_solid,
        "wall_clock_s": wall,
        "s_per_step": wall / n_t,
        "outdir": str(outdir),
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # The plotting scripts rebuild the config to locate the interface, so keep the
    # one this run actually used next to its checkpoints
    with open(outdir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    logger.info(
        "Done in %.1f s (%.4g s/step). liquid_fraction=%.5f mean_interface_x=%.6f m "
        "Nu_hot=%.5f max|v| liquid=%.3f mm/s solid=%.2e mm/s",
        wall,
        summary["s_per_step"],
        summary["liquid_fraction"],
        summary["mean_interface_x"],
        summary["Nu_hot"],
        1e3 * v_liquid,
        1e3 * v_solid,
    )
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run(args)


if __name__ == "__main__":
    main()
