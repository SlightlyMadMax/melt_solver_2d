#!/usr/bin/env python
"""
Water-filled crevasse in an ice slab (manuscript, crevasse application).

A square domain whose bottom wall is held below the melting point and whose top wall is
held above it; the side walls are adiabatic. The ice slab carries a vertical crevasse
filled with water, with a layer of water on top of the slab.

Every numerical parameter a sensitivity study varies is on the command line, so no
source edits are needed between runs, and a run can be continued from a checkpoint.

Examples
--------
    python -m src.examples.crevasse.run --end-time 86400
    python -m src.examples.crevasse.run --nx 201 --ny 201 --dt 0.2 --eps-t 0.1
    python -m src.examples.crevasse.run --no-flow            # conductive baseline
    python -m src.examples.crevasse.run --resume data/.../checkpoint_1728000.npz
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
from src.heat_transfer.energy_ledger import EnergyLedger
from src.heat_transfer.init_values import init_temperature_with_interface
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.parameters.config import ExperimentConfig
from src.utils.boundary_conditions import (
    const_dirichlet_condition,
    const_neumann_condition,
)
from src.utils.nusselt import calculate_nusselt

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("crevasse")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Water-filled crevasse in an ice slab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    case = p.add_argument_group("scenario")
    case.add_argument(
        "--t-top", type=float, default=5.0, help="top wall temperature [degC]"
    )
    case.add_argument(
        "--t-bottom", type=float, default=-10.0, help="bottom wall temperature [degC]"
    )
    case.add_argument(
        "--water-thickness",
        type=float,
        default=0.025,
        help="depth of the water layer above the slab [m]",
    )
    case.add_argument(
        "--crevasse-width", type=float, default=0.025, help="crevasse width [m]"
    )
    case.add_argument(
        "--crevasse-depth", type=float, default=0.15, help="crevasse depth [m]"
    )
    case.add_argument(
        "--no-flow",
        action="store_true",
        help="skip the Navier-Stokes step: conductive baseline",
    )
    case.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint to continue from; the fields are taken from it and the clock "
        "restarts at zero",
    )

    case.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="physical time the resumed state already carries [s]. The runner counts "
        "from zero, so a run stitched together from stages with different time steps "
        "needs this to keep one clock in the energy budget and the summary; the "
        "boundary conditions here are constant, so it changes nothing physically",
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
        help="the same for the penalty term [K]; defaults to --eps-t",
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
        "--heat-convection",
        choices=("deferred", "central-div", "deferred-div"),
        default="deferred",
        help="convective term of the heat equation: first-order upwind with limited "
        "deferred correction, central differences in divergent form d(v u)/dx, or "
        "upwind with limited deferred correction written as face fluxes, which "
        "telescope over the domain like the central ones",
    )
    phys.add_argument(
        "--no-latent-convection",
        action="store_true",
        help="multiply the convective term of the heat equation by c instead of "
        "c + lambda*delta, so that the flow carries sensible heat only",
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
        help="checkpoint interval [s]; 0 keeps only the final state",
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
        "fine-level operator is refreshed",
    )

    misc = p.add_argument_group("misc")
    misc.add_argument("--config", type=Path, default=HERE / "convection" / "config.json")
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
        "conduction" if args.no_flow else "convection",
        f"{g.n_x}x{g.n_y}",
        f"dt{g.dt:g}",
        f"epsT{cfg.delta:g}",
        f"C{penalty_c_of(cfg):.0e}",
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


def crevasse_liquid_depth(u: np.ndarray, cfg: ExperimentConfig) -> float:
    """Depth of the water column on the crevasse axis, from the top of the domain [m].

    The central column is scanned downwards; the unbroken run of liquid nodes tells how
    far the water reaches, which is the quantity the crevasse study is about.
    """
    column = u[:, u.shape[1] // 2] > cfg.u_pt_nd
    depth = 0
    for liquid in reversed(column):
        if not liquid:
            break
        depth += 1
    return depth * cfg.geometry.dy


def speeds(state: SimulationState, cfg: ExperimentConfig) -> tuple[float, float]:
    """Largest speed in the liquid and in the solid [m/s], from the stream function."""
    v_x, v_y = np.empty_like(state.sf), np.empty_like(state.sf)
    calculate_velocity_from_sf(state.sf, v_x, v_y, cfg)
    speed = np.hypot(v_x, v_y) * cfg.v
    solid = state.u < cfg.u_pt_nd
    in_liquid = float(speed[~solid].max()) if (~solid).any() else 0.0
    in_solid = float(speed[solid].max()) if solid.any() else 0.0
    return in_liquid, in_solid


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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    cfg = build_config(args)
    geometry = cfg.geometry
    n_x, n_y, n_t, dt = geometry.n_x, geometry.n_y, geometry.n_t, geometry.dt

    outdir = auto_outdir(args, cfg)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(cfg)
    logger.info(
        "Run: %s grid=%dx%d dt=%g s end=%g s (%d steps) eps_T=%g K eps_flow=%g K "
        "C=%.3g 1/s",
        "conduction" if args.no_flow else "convection",
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

    if args.resume is not None:
        with np.load(args.resume, allow_pickle=True) as d:
            u, sf, w = d["u"].copy(), d["sf"].copy(), d["w"].copy()
            v_x = d["v_x"].copy() if "v_x" in d.files else np.zeros_like(u)
            v_y = d["v_y"].copy() if "v_y" in d.files else np.zeros_like(u)
        if u.shape != (n_y, n_x):
            raise SystemExit(
                f"{args.resume} holds a {u.shape} field, but the run needs {(n_y, n_x)}"
            )
        logger.info("Continuing from %s", args.resume)
    else:
        # The slab with its crevasse: the interface lies deeper on the crevasse axis
        x = np.arange(n_x) * geometry.dx
        slab_top = geometry.height - args.water_thickness
        f = np.where(
            np.abs(x - geometry.width / 2) <= args.crevasse_width / 2,
            slab_top - args.crevasse_depth,
            slab_top,
        )
        u = init_temperature_with_interface(
            cfg=cfg,
            f=f,
            liquid_region_height=args.water_thickness,
            liquid_temp=args.t_top - ABS_ZERO,
            solid_temp=args.t_bottom - ABS_ZERO,
        )
        sf = initialize_stream_function(geometry=geometry, bcs=sf_bcs)
        w = initialize_vorticity(geometry=geometry)
        v_x, v_y = initialize_velocity(geometry=geometry)
        logger.info(
            "Crevasse %.3f m wide and %.3f m deep, water layer %.3f m",
            args.crevasse_width,
            args.crevasse_depth,
            args.water_thickness,
        )

    state = SimulationState(u=u, sf=sf, w=w, v_x=v_x, v_y=v_y)

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
            ledger.record(s.u, s.t + args.time_offset)
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
        calculate_velocity=not args.no_flow,
        save_final=not args.no_save_final,
        save_at=save_at,
        log_at=steps_at_interval(args.log_interval, dt, n_t),
        step_callback=step_callback,
        metrics={
            "liquid_fraction": lambda s: liquid_fraction(s.u, cfg),
            "crevasse_liquid_depth, m": lambda s: crevasse_liquid_depth(s.u, cfg),
            "T_min, degC": lambda s: np.min(s.u * delta_u + u_ref + ABS_ZERO),
            "T_max, degC": lambda s: np.max(s.u * delta_u + u_ref + ABS_ZERO),
        },
    )

    wall_t0 = time.perf_counter()
    runner.run()
    wall = time.perf_counter() - wall_t0

    v_liquid, v_solid = speeds(state, cfg)
    summary = {
        "flow": not args.no_flow,
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
        "heat_convection": args.heat_convection,
        "latent_convection": not args.no_latent_convection,
        "vorticity_bc_order": args.vorticity_bc_order,
        "sf_tolerance": args.sf_tolerance,
        "t_top": args.t_top,
        "t_bottom": args.t_bottom,
        "water_thickness": args.water_thickness,
        "crevasse_width": args.crevasse_width,
        "crevasse_depth": args.crevasse_depth,
        "resumed_from": str(args.resume) if args.resume is not None else None,
        "time_offset": args.time_offset,
        "end_time_absolute": args.time_offset + geometry.end_time,
        "Ra": cfg.rayleigh_number,
        "Pr": cfg.prandtl_number,
        "Ste": cfg.stefan_number,
        "liquid_fraction": liquid_fraction(state.u, cfg),
        "crevasse_liquid_depth": crevasse_liquid_depth(state.u, cfg),
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
        budget_path = ledger.save(outdir / "energy_budget.npz")
        logger.info("Energy budget: %s -> %s", ledger.summary(), budget_path)

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.summary_csv.exists()
        with open(args.summary_csv, "a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(summary))
            if write_header:
                wr.writeheader()
            wr.writerow(summary)

    logger.info(
        "Done in %.1f s (%.4g s/step). liquid_fraction=%.4f crevasse_depth=%.4f m "
        "Nu_top=%.4f max|v| liquid=%.3f mm/s solid=%.2e mm/s",
        wall,
        summary["s_per_step"],
        summary["liquid_fraction"],
        summary["crevasse_liquid_depth"],
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
    run(args)


if __name__ == "__main__":
    main()
