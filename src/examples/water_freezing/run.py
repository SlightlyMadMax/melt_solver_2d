#!/usr/bin/env python
"""
Water freezing in a differentially heated square cavity (Kowalewski & Rebow, 1997).

Driver for the refinement study: every numerical parameter that has to be varied
(grid, time step, smoothing width, penalty coefficient, initial condition) is
exposed on the command line, so no source edits are needed between runs.

Examples
--------
Baseline, cold start, up to t = 2340 s:
    python -m src.examples.water_freezing.run

Grid refinement:
    python -m src.examples.water_freezing.run --nx 101 --ny 101
    python -m src.examples.water_freezing.run --nx 201 --ny 201

Penalty sensitivity:
    python -m src.examples.water_freezing.run --penalty-c 1e5
    python -m src.examples.water_freezing.run --penalty-c 1e7

Warm start (needs a precursor field, see prepare_warm_start.py):
    python -m src.examples.water_freezing.run --start warm

Profiling a short run:
    python -m src.examples.water_freezing.run --end-time 20 --profile
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
from src.heat_transfer.coefficient_smoothing.coefficients import DeltaScheme, StepScheme
from src.heat_transfer.energy_ledger import EnergyLedger
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.parameters.config import ExperimentConfig
from src.utils.boundary_conditions import (
    const_neumann_condition,
    const_dirichlet_condition,
    linear_dirichlet_ramp,
)
from src.utils.nusselt import calculate_nusselt

HERE = Path(__file__).resolve().parent

# Wall temperatures of the benchmark [K]
T_HOT = 283.15  # +10 C, left wall
T_COLD = 263.15  # -10 C, right wall

logger = logging.getLogger("water_freezing")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Water freezing in a square cavity — refinement-study driver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grid = p.add_argument_group("grid and time")
    grid.add_argument("--nx", type=int, default=None, help="grid points in x")
    grid.add_argument("--ny", type=int, default=None, help="grid points in y")
    grid.add_argument("--dt", type=float, default=None, help="time step [s]")
    grid.add_argument(
        "--end-time", type=float, default=2340.0, help="final physical time [s]"
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
        help="bring the penalty up to full strength linearly over this many seconds "
        "of physical time; 0 applies it at full strength from the first step",
    )
    phys.add_argument(
        "--penalty-ramp-mode",
        choices=("linear", "geom"),
        default="linear",
        help="how C grows during --penalty-ramp: proportionally to time, or "
        "geometrically from 1 to its nominal value (equal time per decade)",
    )
    phys.add_argument(
        "--penalty-time-scheme",
        choices=("cn", "dr", "implicit"),
        default="cn",
        help="time discretisation of the penalty term. 'cn' is the published scheme: "
        "explicit on psi^n in the vorticity predictor, implicit with tau/2 in the "
        "stream-function operator. 'implicit' drops the explicit half and gives the "
        "elliptic operator the full tau, i.e. backward Euler on the drag",
    )

    ic = p.add_argument_group("initial condition")
    ic.add_argument(
        "--start",
        choices=["cold", "warm"],
        default="cold",
        help="cold = uniform temperature; warm = precursor steady state",
    )
    ic.add_argument(
        "--cold-start-temp",
        type=float,
        default=0.5,
        help="uniform initial temperature for the cold start [degC]",
    )
    ic.add_argument(
        "--warm-start-file",
        type=Path,
        default=None,
        help="npz with the precursor field; defaults to the grid-matched file "
        "under data/precursors/, then data/precursors/initial_distribution.npz",
    )
    ic.add_argument(
        "--allow-warm-start-interp",
        action="store_true",
        help="permit bilinear interpolation of the precursor field onto a "
        "different grid (contaminates a grid-convergence measurement)",
    )
    ic.add_argument(
        "--cold-wall-ramp",
        type=float,
        default=0.0,
        help="drive the cold wall from its precursor value down to T_COLD linearly "
        "over this many seconds instead of stepping it instantaneously. 0 keeps the "
        "instantaneous step. A short ramp removes the start-up instability that an "
        "instantaneous 10 K step triggers on fine grids",
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
        "--plot-interval",
        type=float,
        default=0.0,
        help="plot interval [s]; 0 disables plotting (default for refinement runs)",
    )
    out.add_argument(
        "--log-interval", type=float, default=60.0, help="log interval [s]"
    )
    out.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="CSV to append the run summary to. Opt-in on purpose: a shared default "
        "collects every throwaway probe alongside the production runs, and telling "
        "them apart afterwards is guesswork. The per-run summary.json is always "
        "written to the output directory regardless",
    )

    out.add_argument(
        "--energy-budget",
        action="store_true",
        help="keep a per-step domain energy budget (wall heat, sensible and latent "
        "enthalpy, defect) and write it to <outdir>/energy_budget.npz",
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
        "of --amg-rebuild-every; the penalty field moves fastest while the solid is "
        "first taking shape, and a hierarchy one step old is already useless there",
    )
    perf.add_argument(
        "--amg-rebuild-every",
        type=int,
        default=1,
        help="rebuild the AMG hierarchy every N time steps; in between only the "
        "fine-level operator is refreshed, which leaves the solution unchanged to "
        "round-off. 1 reproduces the original behaviour",
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


def _amg_stats(navier_solver) -> dict:
    """Hierarchy reuse diagnostics, when the stream-function solver reports them."""
    return getattr(navier_solver.stream_function_solver, "rebuild_stats", {}) or {}


def penalty_c_of(cfg: ExperimentConfig) -> float:
    """Dimensional penalty coefficient C [1/s] implied by cfg.epsilon."""
    return 1.0 / cfg.epsilon**2


def auto_outdir(args: argparse.Namespace, cfg: ExperimentConfig) -> Path:
    if args.outdir is not None:
        return args.outdir
    g = cfg.geometry
    parts = [
        args.start,
        f"{g.n_x}x{g.n_y}",
        f"dt{g.dt:g}",
        f"epsT{cfg.delta:g}",
        f"C{penalty_c_of(cfg):.0e}",
        f"bc{args.vorticity_bc_order}",
    ]
    if args.tag:
        parts.append(args.tag)
    return HERE / "data" / "scratch" / "_".join(parts)


# ----------------------------------------------------------------------------
# Initial conditions
# ----------------------------------------------------------------------------


def _bilinear_resample(field: np.ndarray, n_y: int, n_x: int) -> np.ndarray:
    """Resample a field given on a uniform grid onto a different uniform grid."""
    src_y, src_x = field.shape
    ys = np.linspace(0.0, src_y - 1, n_y)
    xs = np.linspace(0.0, src_x - 1, n_x)

    j0 = np.clip(np.floor(ys).astype(int), 0, src_y - 2)
    i0 = np.clip(np.floor(xs).astype(int), 0, src_x - 2)
    ty = (ys - j0)[:, None]
    tx = (xs - i0)[None, :]

    f00 = field[np.ix_(j0, i0)]
    f01 = field[np.ix_(j0, i0 + 1)]
    f10 = field[np.ix_(j0 + 1, i0)]
    f11 = field[np.ix_(j0 + 1, i0 + 1)]

    return (
        f00 * (1 - ty) * (1 - tx)
        + f01 * (1 - ty) * tx
        + f10 * ty * (1 - tx)
        + f11 * ty * tx
    )


def resolve_warm_start_file(args: argparse.Namespace, cfg: ExperimentConfig) -> Path:
    if args.warm_start_file is not None:
        return args.warm_start_file
    g = cfg.geometry
    candidates = [
        HERE / "data" / "precursors" / f"steady_{g.n_x}x{g.n_y}.npz",
        HERE / "data" / "precursors" / "initial_distribution.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "No warm-start field found. Looked for:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nGenerate one with prepare_warm_start.py, or pass --warm-start-file."
    )


def make_initial_state(
    args: argparse.Namespace, cfg: ExperimentConfig, u_bcs, sf_bcs
) -> SimulationState:
    geometry = cfg.geometry

    if args.start == "cold":
        u = init_temperature(
            cfg=cfg,
            bcs=u_bcs,
            shape=DomainShape.UNIFORM_LIQUID,
            liquid_temp=args.cold_start_temp - ABS_ZERO,
        )
        sf = initialize_stream_function(geometry=geometry, bcs=sf_bcs)
        w = initialize_vorticity(geometry=geometry)
        v_x, v_y = initialize_velocity(geometry=geometry)
        logger.info("Cold start: uniform T = %.3f degC", args.cold_start_temp)
        return SimulationState(u=u, sf=sf, w=w, v_x=v_x, v_y=v_y)

    path = resolve_warm_start_file(args, cfg)
    with np.load(path, allow_pickle=True) as d:
        u, sf, w = d["u"], d["sf"], d["w"]
        v_x = d["v_x"] if "v_x" in d.files else np.zeros_like(u)
        v_y = d["v_y"] if "v_y" in d.files else np.zeros_like(u)

    target = (geometry.n_y, geometry.n_x)
    if u.shape != target:
        if not args.allow_warm_start_interp:
            raise SystemExit(
                f"Warm-start field {path} is {u.shape}, but the run needs {target}.\n"
                "The precursor is a steady state of the convection problem and should "
                "be recomputed at the target resolution:\n"
                f"    python -m src.examples.water_freezing.prepare_warm_start "
                f"--nx {geometry.n_x} --ny {geometry.n_y}\n"
                "Pass --allow-warm-start-interp to interpolate instead (this makes the "
                "initial condition resolution-dependent and will contaminate a grid "
                "convergence measurement)."
            )
        logger.warning(
            "Interpolating warm-start field %s -> %s. The initial condition is now "
            "resolution-dependent; do not use these runs for grid convergence.",
            u.shape,
            target,
        )
        u, sf, w, v_x, v_y = (
            _bilinear_resample(a, *target) for a in (u, sf, w, v_x, v_y)
        )

    logger.info("Warm start from %s", path)
    return SimulationState(
        u=np.ascontiguousarray(u),
        sf=np.ascontiguousarray(sf),
        w=np.ascontiguousarray(w),
        v_x=np.ascontiguousarray(v_x),
        v_y=np.ascontiguousarray(v_y),
    )


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


def ice_area_fraction(u: np.ndarray, u_pt_nd: float) -> float:
    """Area fraction of the domain below the phase-change temperature."""
    return float(np.mean(u < u_pt_nd))


def mean_interface_x(u: np.ndarray, cfg: ExperimentConfig) -> float:
    """
    Mean horizontal position of the T = T_m isotherm [m].

    Ice grows from the cold (right) wall, so each row is scanned from the right for
    the first crossing and the position is refined by linear interpolation.
    """
    u_pt = cfg.u_pt_nd
    dx = cfg.geometry.dx
    n_y, n_x = u.shape
    positions = []
    for j in range(n_y):
        row = u[j]
        solid = row < u_pt
        if solid.all():
            positions.append(0.0)  # row frozen all the way across
            continue
        if not solid[n_x - 1]:
            continue  # no ice attached to the cold wall in this row
        # Walk left from the cold wall to the first liquid node. It exists because
        # the row is not fully solid, so i >= 0 and i + 1 stays in range.
        i = n_x - 1
        while solid[i]:
            i -= 1
        a, b = row[i], row[i + 1]
        frac = 0.0 if a == b else (u_pt - a) / (b - a)
        positions.append((i + frac) * dx)
    return float(np.mean(positions)) if positions else float("nan")


def max_speed_in_solid(state: SimulationState, cfg: ExperimentConfig) -> float:
    """Largest residual dimensionless speed inside the solid phase."""
    if state.v_x is None or state.v_y is None:
        return float("nan")
    mask = state.u < cfg.u_pt_nd
    if not mask.any():
        return 0.0
    speed = np.hypot(state.v_x, state.v_y)
    return float(speed[mask].max())


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
        "Run: start=%s grid=%dx%d dt=%g s end=%g s (%d steps) "
        "eps_T=%g K eps_flow=%g K C=%.3g 1/s",
        args.start,
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

    cold_nd = (T_COLD - u_ref) / delta_u
    if args.cold_wall_ramp > 0.0:
        # Precursor cold wall sits at the phase-change temperature.
        right_bc = linear_dirichlet_ramp(
            n_y,
            start_value=(cfg.material_props.u_pt - u_ref) / delta_u,
            end_value=cold_nd,
            duration=args.cold_wall_ramp,
        )
        logger.info("Cold wall ramped to %.1f K over %g s", T_COLD, args.cold_wall_ramp)
    else:
        right_bc = const_dirichlet_condition(n_y, value=cold_nd)

    u_bcs = BoundaryConditions(
        top=const_neumann_condition(n_x, value=0.0),
        right=right_bc,
        bottom=const_neumann_condition(n_x, value=0.0),
        left=const_dirichlet_condition(n_y, value=(T_HOT - u_ref) / delta_u),
    )
    sf_bcs = BoundaryConditions(
        top=const_dirichlet_condition(n_x, value=0.0),
        right=const_dirichlet_condition(n_y, value=0.0),
        bottom=const_dirichlet_condition(n_x, value=0.0),
        left=const_dirichlet_condition(n_y, value=0.0),
    )

    state = make_initial_state(args, cfg, u_bcs, sf_bcs)

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
        calculate_velocity=True,
        save_final=not args.no_save_final,
        save_at=save_at,
        plot_at=steps_at_interval(args.plot_interval, dt, n_t),
        log_at=steps_at_interval(args.log_interval, dt, n_t),
        metrics={
            "T_max, °C": lambda s: np.max(s.u * delta_u + u_ref + ABS_ZERO),
            "T_min, °C": lambda s: np.min(s.u * delta_u + u_ref + ABS_ZERO),
            "ice_fraction": lambda s: ice_area_fraction(s.u, cfg.u_pt_nd),
            "Nu_hot": lambda s: calculate_nusselt(u=s.u, cfg=cfg, wall="left"),
        },
        step_callback=step_callback,
    )

    wall_t0 = time.perf_counter()
    runner.run()
    wall = time.perf_counter() - wall_t0

    summary = {
        "start": args.start,
        "n_x": n_x,
        "n_y": n_y,
        "dx": geometry.dx,
        "dy": geometry.dy,
        "dt": dt,
        "end_time": geometry.end_time,
        "n_t": n_t,
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
        "ice_fraction": ice_area_fraction(state.u, cfg.u_pt_nd),
        "mean_interface_x": mean_interface_x(state.u, cfg),
        "Nu_hot": calculate_nusselt(u=state.u, cfg=cfg, wall="left"),
        "Nu_cold": calculate_nusselt(u=state.u, cfg=cfg, wall="right"),
        "max_speed_in_solid": max_speed_in_solid(state, cfg),
        "cold_wall_ramp": args.cold_wall_ramp,
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
    else:
        csv_path = None

    logger.info(
        "Done in %.1f s (%.4g s/step). ice_fraction=%.6f mean_interface_x=%.6f m "
        "Nu_hot=%.5f max|v| in solid=%.3e",
        wall,
        summary["s_per_step"],
        summary["ice_fraction"],
        summary["mean_interface_x"],
        summary["Nu_hot"],
        summary["max_speed_in_solid"],
    )
    if csv_path is not None:
        logger.info("Summary appended to %s", csv_path)
    else:
        logger.info("Summary written to %s", outdir / "summary.json")
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
