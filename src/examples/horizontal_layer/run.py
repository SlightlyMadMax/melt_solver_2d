#!/usr/bin/env python
"""
Melting and freezing of a horizontal water-ice layer (manuscript, section 5.1).

A 20 cm wide, 5 cm high domain. The top wall is held at +5 C and the bottom wall at
-5 C; the side walls are adiabatic. Two scenarios start from a single phase:

    melting   the domain is ice at -5 C and melts from the top
    freezing  the domain is water at +5 C and ice grows from the bottom

With --no-flow the Navier-Stokes step is skipped and the problem reduces to the
conductive Stefan problem, the baseline the manuscript compares against.

Every numerical parameter that a sensitivity study varies is exposed on the command
line, so no source edits are needed between runs.

Examples
--------
Manuscript runs, 24 h:
    python -m src.examples.horizontal_layer.run --case melting
    python -m src.examples.horizontal_layer.run --case freezing

Conductive baseline:
    python -m src.examples.horizontal_layer.run --case freezing --no-flow

Short timing run:
    python -m src.examples.horizontal_layer.run --case freezing --end-time 600
"""

import argparse
import csv
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
    initialize_vorticity,
    initialize_velocity,
)
from src.fluid_dynamics.solvers import VorticitySolverName, StreamFunctionSolverName
from src.fluid_dynamics.solvers.bc_correction_solver_factory import BCCorrectionNVSolver
from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import PenaltyTermForm
from src.fluid_dynamics.utils import calculate_velocity_from_sf
from src.heat_transfer.coefficient_smoothing.coefficients import DeltaScheme, StepScheme
from src.heat_transfer.energy_ledger import EnergyLedger
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.parameters.config import ExperimentConfig
from src.utils.boundary_conditions import (
    const_neumann_condition,
    const_dirichlet_condition,
)
from src.utils.nusselt import calculate_nusselt

HERE = Path(__file__).resolve().parent

logger = logging.getLogger("horizontal_layer")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Melting / freezing of a horizontal water-ice layer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    case = p.add_argument_group("scenario")
    case.add_argument(
        "--case",
        choices=("melting", "freezing"),
        required=True,
        help="melting: start from ice; freezing: start from water",
    )
    case.add_argument(
        "--no-flow",
        action="store_true",
        help="skip the Navier-Stokes step: conductive Stefan problem",
    )
    case.add_argument("--t-top", type=float, default=5.0, help="top wall temperature [degC]")
    case.add_argument(
        "--t-bottom", type=float, default=-5.0, help="bottom wall temperature [degC]"
    )
    case.add_argument(
        "--initial-temp",
        type=float,
        default=None,
        help="uniform initial temperature [degC]; defaults to --t-bottom for melting "
        "(ice) and --t-top for freezing (water)",
    )

    grid = p.add_argument_group("grid and time")
    grid.add_argument("--nx", type=int, default=None, help="grid points in x")
    grid.add_argument("--ny", type=int, default=None, help="grid points in y")
    grid.add_argument("--dt", type=float, default=None, help="time step [s]")
    grid.add_argument(
        "--end-time", type=float, default=86400.0, help="final physical time [s]"
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
        "defaults to --eps-t when not given",
    )
    phys.add_argument(
        "--penalty-c",
        type=float,
        default=None,
        help="penalty coefficient C [1/s]; stored in the config as epsilon = 1/sqrt(C)",
    )
    phys.add_argument(
        "--vorticity-bc-order",
        type=int,
        choices=(1, 2),
        default=2,
        help="order of the wall vorticity condition: 1 = Thom, 2 = Woods/Jensen",
    )
    phys.add_argument(
        "--penalty-form",
        choices=[f.name.lower() for f in PenaltyTermForm],
        default="quadratic",
        help="functional form of the penalty term",
    )
    phys.add_argument(
        "--penalty-ramp",
        type=float,
        default=0.0,
        help="bring the penalty up to full strength over this many seconds of physical "
        "time; 0 applies it at full strength from the first step",
    )
    phys.add_argument(
        "--penalty-ramp-mode",
        choices=("linear", "geom"),
        default="linear",
        help="how C grows during --penalty-ramp",
    )
    phys.add_argument(
        "--penalty-time-scheme",
        choices=("cn", "dr", "implicit"),
        default="cn",
        help="time discretisation of the penalty term; 'cn' is the published scheme",
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
        "--save-interval",
        type=float,
        default=0.0,
        help="checkpoint interval [s]; 0 disables intermediate checkpoints",
    )
    out.add_argument(
        "--no-save-final", action="store_true", help="do not save the final state"
    )
    out.add_argument(
        "--log-interval", type=float, default=600.0, help="log interval [s]"
    )
    out.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="CSV to append the run summary to; summary.json is always written",
    )
    out.add_argument(
        "--energy-budget",
        action="store_true",
        help="keep a per-step domain energy budget and write it to "
        "<outdir>/energy_budget.npz",
    )

    num = p.add_argument_group("linear solver")
    num.add_argument(
        "--sf-tolerance",
        type=float,
        default=1e-6,
        help="convergence tolerance of the stream-function elliptic solve",
    )

    perf = p.add_argument_group("performance")
    perf.add_argument(
        "--amg-rebuild-warmup",
        type=int,
        default=0,
        help="rebuild the AMG hierarchy on every one of the first N steps regardless "
        "of --amg-rebuild-every",
    )
    perf.add_argument(
        "--amg-rebuild-every",
        type=int,
        default=1,
        help="rebuild the AMG hierarchy every N time steps; in between only the "
        "fine-level operator is refreshed. 1 reproduces the original behaviour",
    )

    misc = p.add_argument_group("misc")
    misc.add_argument("--config", type=Path, default=HERE / "config.json")
    misc.add_argument(
        "--profile", action="store_true", help="run under cProfile and dump stats"
    )
    misc.add_argument(
        "--profile-out",
        type=Path,
        default=None,
        help="path for the .prof file (defaults to <outdir>/profile.prof)",
    )
    misc.add_argument("--quiet", action="store_true", help="only warnings and errors")

    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """
    Apply the CLI overrides to the JSON payload *before* validation.

    ExperimentConfig caches derived quantities (scaled_grid_steps, v, Pe, ...) via
    cached_property, so mutating an already-constructed instance would leave those
    stale. Overriding the raw dict avoids the problem entirely.
    """
    with open(args.config, "r", encoding="utf-8") as f:
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
    # Keep dt exact: end_time is snapped to a whole number of steps.
    geom["end_time"] = n_t * dt
    geom["n_t"] = n_t

    if args.eps_t is not None:
        data["delta"] = args.eps_t
    data["delta_flow"] = (
        args.eps_flow
        if args.eps_flow is not None
        else (args.eps_t if args.eps_t is not None else data.get("delta_flow"))
    )

    if args.penalty_c is not None:
        # The solver builds the dimensionless penalty as l / (epsilon**2 * v),
        # which corresponds to a dimensional C = 1 / epsilon**2 [1/s].
        data["epsilon"] = 1.0 / math.sqrt(args.penalty_c)

    return ExperimentConfig.model_validate(data)


def penalty_c_of(cfg: ExperimentConfig) -> float:
    """Dimensional penalty coefficient C [1/s] implied by cfg.epsilon."""
    return 1.0 / cfg.epsilon**2


def _amg_stats(navier_solver) -> dict:
    """Hierarchy reuse diagnostics, when the stream-function solver reports them."""
    sf_solver = getattr(navier_solver, "stream_function_solver", None)
    return getattr(sf_solver, "rebuild_stats", {}) or {}


def auto_outdir(args: argparse.Namespace, cfg: ExperimentConfig) -> Path:
    if args.outdir is not None:
        return args.outdir
    g = cfg.geometry
    parts = [
        args.case + ("_noflow" if args.no_flow else ""),
        f"{g.n_x}x{g.n_y}",
        f"dt{g.dt:g}",
        f"epsT{cfg.delta:g}",
        f"C{penalty_c_of(cfg):.0e}",
        f"bc{args.vorticity_bc_order}",
    ]
    if args.tag:
        parts.append(args.tag)
    return HERE / "data" / "scratch" / "_".join(parts)


class NoFlow:
    """Stands in for the Navier-Stokes solver in the conductive baseline."""

    stream_function_solver = None

    def solve(self, w, sf, u, delta, time):
        return sf, w


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


def liquid_fraction(u: np.ndarray, cfg: ExperimentConfig) -> float:
    """Area fraction of the domain above the phase-change temperature."""
    return float(np.mean(u > cfg.u_pt_nd))


def interface_heights(u: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """
    Height of the T = T_m isotherm in every column [m], nan where a column holds a
    single phase.

    Each column is scanned from the bottom for its own first crossing and refined by
    linear interpolation. The ice sits below the water in both scenarios, so the first
    crossing from the bottom is the ice-water interface.
    """
    u_pt = cfg.u_pt_nd
    dy = cfg.geometry.dy
    diff = u - u_pt
    crossing = diff[:-1] * diff[1:] < 0  # (n_y - 1, n_x)
    heights = np.full(u.shape[1], np.nan)
    has = crossing.any(axis=0)
    j = np.argmax(crossing, axis=0)
    cols = np.nonzero(has)[0]
    j = j[cols]
    a, b = u[j, cols], u[j + 1, cols]
    heights[cols] = (j + (u_pt - a) / (b - a)) * dy
    return heights


def mean_interface_y(u: np.ndarray, cfg: ExperimentConfig) -> float:
    h = interface_heights(u, cfg)
    return float(np.nanmean(h)) if np.isfinite(h).any() else float("nan")


def interface_amplitude(u: np.ndarray, cfg: ExperimentConfig) -> float:
    """Peak-to-peak height of the interface [m]."""
    h = interface_heights(u, cfg)
    return float(np.nanmax(h) - np.nanmin(h)) if np.isfinite(h).any() else float("nan")


def speeds(state: SimulationState, cfg: ExperimentConfig) -> tuple[float, float]:
    """Largest speed in the liquid and in the solid [m/s], from the stream function.

    The velocity is not advanced with the state (it is not needed by the solvers), so
    it is reconstructed here with the same central differences the solvers use.
    """
    v_x, v_y = np.empty_like(state.sf), np.empty_like(state.sf)
    calculate_velocity_from_sf(state.sf, v_x, v_y, cfg)
    speed = np.hypot(v_x, v_y) * cfg.v
    solid = state.u < cfg.u_pt_nd
    in_liquid = float(speed[~solid].max()) if (~solid).any() else 0.0
    in_solid = float(speed[solid].max()) if solid.any() else 0.0
    return in_liquid, in_solid


def conductive_steady_interface(args: argparse.Namespace, cfg: ExperimentConfig) -> float:
    """Height of the planar interface of the steady conductive problem [m].

    Equal heat fluxes through ice and water: k_s (T_m - T_b) / h = k_l (T_t - T_m) / (H - h).
    """
    props = cfg.material_props
    t_m = props.u_pt + ABS_ZERO
    q_s = props.thermal_conductivity_solid * (t_m - args.t_bottom)
    q_l = props.thermal_conductivity_liquid * (args.t_top - t_m)
    return cfg.geometry.height * q_s / (q_s + q_l)


# ----------------------------------------------------------------------------
# Scheduling
# ----------------------------------------------------------------------------


def steps_at_interval(interval: float, dt: float, n_t: int) -> set[int]:
    """
    Step indices closest to multiples of `interval`.

    Integer arithmetic on purpose: `n * dt % interval == 0` silently misses events
    because n * dt is almost never an exact multiple of the interval in floating
    point.
    """
    if interval is None or interval <= 0.0:
        return set()
    stride = max(1, int(round(interval / dt)))
    return set(range(stride, n_t + 1, stride))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    cfg = build_config(args)
    geometry: DomainGeometry = cfg.geometry
    n_x, n_y, n_t, dt = geometry.n_x, geometry.n_y, geometry.n_t, geometry.dt

    outdir = auto_outdir(args, cfg)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(cfg)
    logger.info(
        "Run: case=%s flow=%s grid=%dx%d dt=%g s end=%g s (%d steps) "
        "eps_T=%g K eps_flow=%g K C=%.3g 1/s",
        args.case,
        not args.no_flow,
        n_x,
        n_y,
        dt,
        geometry.end_time,
        n_t,
        cfg.delta,
        cfg.delta_flow,
        penalty_c_of(cfg),
    )
    logger.info("Output: %s", outdir)

    delta_u, u_ref = cfg.delta_u, cfg.u_ref

    def nd(t_celsius: float) -> float:
        return (t_celsius - ABS_ZERO - u_ref) / delta_u

    u_bcs = BoundaryConditions(
        top=const_dirichlet_condition(n_x, value=nd(args.t_top)),
        right=const_neumann_condition(n_y, value=0.0),
        bottom=const_dirichlet_condition(n_x, value=nd(args.t_bottom)),
        left=const_neumann_condition(n_y, value=0.0),
    )
    sf_bcs = BoundaryConditions(
        top=const_dirichlet_condition(n_x, value=0.0),
        right=const_dirichlet_condition(n_y, value=0.0),
        bottom=const_dirichlet_condition(n_x, value=0.0),
        left=const_dirichlet_condition(n_y, value=0.0),
    )

    if args.case == "melting":
        t0 = args.t_bottom if args.initial_temp is None else args.initial_temp
        u = init_temperature(
            cfg=cfg, bcs=u_bcs, shape=DomainShape.UNIFORM_SOLID, solid_temp=t0 - ABS_ZERO
        )
    else:
        t0 = args.t_top if args.initial_temp is None else args.initial_temp
        u = init_temperature(
            cfg=cfg, bcs=u_bcs, shape=DomainShape.UNIFORM_LIQUID, liquid_temp=t0 - ABS_ZERO
        )
    logger.info("Initial state: uniform %.2f degC (%s)", t0, args.case)

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
        convective_term_form=ConvectiveTermForm.DEFERRED_CORRECTION,
        step_scheme=StepScheme.ERF,
        delta_scheme=DeltaScheme.GAUSS,
        k_face_method=KFaceMethod.FROM_TEMP,
    )

    if args.no_flow:
        navier_solver = NoFlow()
    else:
        navier_solver = BCCorrectionNVSolver(
            cfg=cfg,
            sf_bcs=sf_bcs,
            sf_max_iters=(n_y - 2) * (n_x - 2),
            sf_tolerance=args.sf_tolerance,
            convective_term_form=ConvectiveTermForm.DIVERGENT_CENTRAL,
            penalty_term_form=PenaltyTermForm[args.penalty_form.upper()],
            vorticity_solver_name=VorticitySolverName.PEACEMAN_RACHFORD,
            stream_function_solver_name=StreamFunctionSolverName.AMG,
            vorticity_bc_order=args.vorticity_bc_order,
            sf_solver_kwargs={
                "rebuild_every": args.amg_rebuild_every,
                "warmup_solves": args.amg_rebuild_warmup,
            },
            penalty_time_scheme=args.penalty_time_scheme,
            penalty_ramp=args.penalty_ramp,
            penalty_ramp_mode=args.penalty_ramp_mode,
        )

    ledger = None
    step_callback = None
    save_at = steps_at_interval(args.save_interval, dt, n_t)
    if args.energy_budget:
        ledger = EnergyLedger(cfg, heat_solver)
        ledger.prime(state.u)

        def step_callback(s: SimulationState) -> None:
            ledger.record(s.u, s.t)
            # The runner writes the fields right after this; writing the budget
            # alongside them leaves both at the same step if the run dies later
            if s.n in save_at:
                ledger.save(outdir / "energy_budget.npz")

    runner = ExperimentRunner(
        cfg=cfg,
        state=state,
        heat_solver=heat_solver,
        navier_solver=navier_solver,
        logger=logger,
        checkpoints_dir=outdir,
        # The solvers work from the stream function alone; the velocity is rebuilt
        # from it where a diagnostic needs it, as in the runs behind the manuscript
        calculate_velocity=False,
        save_final=not args.no_save_final,
        save_at=save_at,
        log_at=steps_at_interval(args.log_interval, dt, n_t),
        metrics={
            "liquid_fraction": lambda s: liquid_fraction(s.u, cfg),
            "mean_interface_y, m": lambda s: mean_interface_y(s.u, cfg),
            "Nu_top": lambda s: calculate_nusselt(u=s.u, cfg=cfg, wall="top"),
            "max|v| in liquid, mm/s": lambda s: 1e3 * speeds(s, cfg)[0],
        },
        step_callback=step_callback,
    )

    wall_t0 = time.perf_counter()
    runner.run()
    wall = time.perf_counter() - wall_t0

    v_liquid, v_solid = speeds(state, cfg)
    summary = {
        "case": args.case,
        "flow": not args.no_flow,
        "n_x": n_x,
        "n_y": n_y,
        "dx": geometry.dx,
        "dy": geometry.dy,
        "dt": dt,
        "end_time": geometry.end_time,
        "n_t": n_t,
        "t_top": args.t_top,
        "t_bottom": args.t_bottom,
        "initial_temp": t0,
        "eps_T": cfg.delta,
        "eps_flow": cfg.delta_flow,
        "penalty_C": penalty_c_of(cfg),
        "penalty_form": args.penalty_form,
        "penalty_time_scheme": args.penalty_time_scheme,
        "penalty_ramp": args.penalty_ramp,
        "penalty_ramp_mode": args.penalty_ramp_mode,
        "vorticity_bc_order": args.vorticity_bc_order,
        "sf_tolerance": args.sf_tolerance,
        "Ra": cfg.rayleigh_number,
        "Pr": cfg.prandtl_number,
        "Ste": cfg.stefan_number,
        "liquid_fraction": liquid_fraction(state.u, cfg),
        "mean_interface_y": mean_interface_y(state.u, cfg),
        "interface_amplitude": interface_amplitude(state.u, cfg),
        "conductive_steady_interface": conductive_steady_interface(args, cfg),
        "Nu_top": calculate_nusselt(u=state.u, cfg=cfg, wall="top"),
        "Nu_bottom": calculate_nusselt(u=state.u, cfg=cfg, wall="bottom"),
        "max_speed_liquid": v_liquid,
        "max_speed_solid": v_solid,
        "amg_rebuild_every": args.amg_rebuild_every,
        "amg_rebuild_warmup": args.amg_rebuild_warmup,
        "amg_rebuilds": _amg_stats(navier_solver).get("rebuilds"),
        "amg_mean_reuse": _amg_stats(navier_solver).get("mean_reuse"),
        "wall_clock_s": wall,
        "s_per_step": wall / n_t,
        "outdir": str(outdir),
    }

    if ledger is not None:
        # Kept out of the summary so the CSV header stays the same across runs
        budget_path = ledger.save(outdir / "energy_budget.npz")
        logger.info("Energy budget: %s -> %s", ledger.summary(), budget_path)

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.summary_csv is not None:
        csv_path = args.summary_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(summary))
            if write_header:
                wr.writeheader()
            wr.writerow(summary)

    logger.info(
        "Done in %.1f s (%.4g s/step). mean_interface_y=%.6f m amplitude=%.2f mm "
        "liquid_fraction=%.4f Nu_top=%.4f max|v| liquid=%.3f mm/s solid=%.2e mm/s",
        wall,
        summary["s_per_step"],
        summary["mean_interface_y"],
        1e3 * summary["interface_amplitude"],
        summary["liquid_fraction"],
        summary["Nu_top"],
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

    if not args.profile:
        run(args)
        return

    import cProfile
    import pstats

    cfg_preview = build_config(args)
    prof_out = args.profile_out or (auto_outdir(args, cfg_preview) / "profile.prof")
    prof_out.parent.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run(args)
    finally:
        profiler.disable()
        profiler.dump_stats(str(prof_out))
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        print(f"\n=== cProfile (cumulative, top 35) -> {prof_out} ===")
        stats.print_stats(35)


if __name__ == "__main__":
    main()
