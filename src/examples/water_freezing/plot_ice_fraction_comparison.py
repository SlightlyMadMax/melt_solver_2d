#!/usr/bin/env python
"""
Ice fraction in time: cold and warm start, published and new scheme, against experiment.

The computed curves come from the energy budget of each run: the latent energy divided
by the latent heat of the whole cavity is the liquid fraction, so

    f_ice(t) = 1 - E_lat(t) / (rho L * area of the interior nodes).

This is the smoothed liquid fraction the model actually carries, and it agrees with the
sharp count of nodes below the melting point (logged as ice_fraction) to about 0.005.

The measured points are the interface profiles of run #4 digitised from Figure 4 of
Kowalewski & Rebow (1997); ice_fraction_from_figure4.py turns them into fractions. Their
bars are the 5-8% run-to-run reproducibility quoted in the paper, taken as 8% of the ice
layer (--error-mode thickness) or, as an alternative reading, of the plotted front
position (--error-mode position).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# numpy renamed it in 2.0; the project still runs on both
trapezoid = getattr(np, "trapezoid", None) or np.trapz

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
    }
)

BASELINE = "151x151_dt0.02_e0.1_ef0.1_C1e+06"


def runs_of(family: str, baseline: str):
    """The two computed curves, warm first because both digitised runs started warm."""
    return [
        ("computed, warm start", DATA / "warm_start" / family / baseline, "#3b0f70", "-"),
        ("computed, cold start", DATA / "cold_start" / family / baseline, "#1f9e89", "--"),
    ]


PIV_CURVE = DATA / "inputs" / "piv_boundary.npz"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--experiment", type=Path, nargs="+",
                   default=[DATA / "inputs" / "figure4_extracted" / "ice_fraction.csv",
                            DATA / "inputs" / "figure4_run8_exatracted" / "ice_fraction.csv"],
                   help="outputs of ice_fraction_from_figure4.py")
    p.add_argument("--experiment-labels", nargs="+",
                   default=["run #4, external air", "run #8, external water"])
    p.add_argument("--markers", nargs="+", default=["o", "s"])
    p.add_argument("--error-mode", choices=("thickness", "position", "none"),
                   default="thickness",
                   help="what the quoted 5-8% is taken to apply to")
    p.add_argument("--family", default="star_sigma05_t3000",
                   help="sub-directory of data/<start>_start holding the runs to draw")
    p.add_argument("--baseline", default=BASELINE, help="run directory inside --family")
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--out", type=Path,
                   default=HERE / "graphs" / "new_scheme" / "ice_fraction_validation.png")
    p.add_argument("--stride", type=int, default=50, help="plot every Nth recorded step")
    p.add_argument("--run1", action=argparse.BooleanOptionalAction, default=True,
                   help="also show the single point of run #1, the front of Figure 1")
    p.add_argument("--run1-curve", type=Path, default=PIV_CURVE,
                   help="front of run #1, x and y in metres")
    p.add_argument("--run1-time", type=float, default=2340.0)
    p.add_argument("--reproducibility", type=float, default=0.08,
                   help="quoted run-to-run reproducibility, used for the bar of run #1")
    return p.parse_args(argv)


def latent_heat_per_volume(config: Path) -> float:
    with open(config, encoding="utf-8") as f:
        props = json.load(f)["material_props"]
    # MaterialProperties.volumetric_latent_heat
    return props["density_liquid"] * props["specific_latent_heat"]


def cavity_width(config: Path) -> float:
    with open(config, encoding="utf-8") as f:
        return json.load(f)["geometry"]["width"]


def run1_fraction(path: Path, width: float) -> float:
    """Ice fraction of the front of Figure 1, stored as x(y) in metres."""
    with np.load(path) as d:
        x, y = np.asarray(d["x"]), np.asarray(d["y"])
    order = np.argsort(y)
    x, y = x[order], y[order]
    return float(trapezoid(1.0 - x / width, y) / (y[-1] - y[0]))


def ice_series(run_dir: Path, latent: float) -> tuple[np.ndarray, np.ndarray]:
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        s = json.load(f)
    a = np.load(run_dir / "energy_budget.npz")
    # the ledger sums over interior nodes only, each carrying one cell
    area = (s["n_x"] - 2) * (s["n_y"] - 2) * s["dx"] * s["dy"]
    return a["t"], 1.0 - a["E_lat"] / (latent * area)


def main(argv=None) -> None:
    args = parse_args(argv)
    latent = latent_heat_per_volume(args.config)

    fig, ax = plt.subplots(figsize=(6.3, 3.4), constrained_layout=True)
    for label, run_dir, color, style in runs_of(args.family, args.baseline):
        t, ice = ice_series(run_dir, latent)
        ax.plot(t[:: args.stride], ice[:: args.stride], color=color, ls=style, label=label)

    shades = ["k", "0.45"]
    for path, label, marker, shade in zip(args.experiment, args.experiment_labels,
                                          args.markers, shades):
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        t_exp = np.array([float(r["t_s"]) for r in rows])
        f_exp = np.array([float(r["ice_fraction"]) for r in rows])
        err = None
        if args.error_mode != "none":
            err = np.array([float(r[f"err_{args.error_mode}"]) for r in rows])
        ax.errorbar(t_exp, f_exp, yerr=err, fmt=marker, ms=4.5, color=shade, mfc="white",
                    mew=1.1, elinewidth=0.9, capsize=2.5, zorder=5, label=label)

    f_run1 = None
    if args.run1:
        f_run1 = run1_fraction(args.run1_curve, cavity_width(args.config))
        e1 = args.reproducibility * f_run1 if args.error_mode == "thickness" else (
            args.reproducibility * (1.0 - f_run1) if args.error_mode == "position" else None)
        ax.errorbar([args.run1_time], [f_run1], yerr=None if e1 is None else [[e1], [e1]],
                    fmt="s", ms=4.5, color="k", mfc="0.55", mew=1.1, elinewidth=0.9,
                    capsize=2.5, zorder=5, label="experiment, run #1")

    ax.set_xlabel(r"$t$, s")
    ax.set_ylabel("ice fraction")
    ax.set_xlim(0.0, max(t_exp.max(), 2340.0) * 1.02)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"-> {args.out}")
    for label, run_dir, _, _ in runs_of(args.family, args.baseline):
        t, ice = ice_series(run_dir, latent)
        at = [f"{te:.0f} s: {ice[np.searchsorted(t, te) - 1]:.4f}"
              for te in t_exp if te <= t[-1]]
        print(f"  {label:<22} " + ", ".join(at))
    for path, label in zip(args.experiment, args.experiment_labels):
        with open(path, encoding="utf-8") as f:
            rr = list(csv.DictReader(f))
        print(f"  {label:<22} " + ", ".join(
            f"{float(r['t_s']):.0f} s: {float(r['ice_fraction']):.4f}" for r in rr))
    if f_run1 is not None:
        print(f"  run #1      {args.run1_time:.0f} s: {f_run1:.4f}")


if __name__ == "__main__":
    main()
