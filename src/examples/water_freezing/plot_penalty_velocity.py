#!/usr/bin/env python
"""
Largest speed left inside the ice against the penalty coefficient, at t = 2340 s,
with the largest speed in the water of the adopted run for scale.

    python -m src.examples.water_freezing.plot_penalty_velocity
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.parameters.config import ExperimentConfig  # noqa: E402
from src.fluid_dynamics.utils import calculate_velocity_from_sf  # noqa: E402

DATA = HERE / "data"

mpl.rcParams.update(
    {
        "font.size": 12, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8.5,
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "lines.linewidth": 1.5, "figure.dpi": 300,
    }
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--family", type=Path, default=DATA / "warm_start" / "star_sigma05")
    p.add_argument("--out", type=Path,
                   default=HERE / "graphs" / "new_scheme" / "penalty_velocity.png")
    p.add_argument("--time", type=float, default=2340.0)
    return p.parse_args(argv)


def load(family: Path):
    runs = {}
    for d in sorted(glob.glob(str(family / "*"))):
        sj = os.path.join(d, "summary.json")
        if not os.path.exists(sj):
            continue
        s = json.load(open(sj, encoding="utf-8"))
        key = (s["n_x"], round(s["dt"], 4), round(s["eps_T"], 4), round(s["penalty_C"]))
        runs.setdefault(key, (s, Path(d)))
    return runs


def liquid_speed(s, d: Path, time: float) -> float:
    """Largest speed in the water of that run, from the stored stream function."""
    data = json.load(open(HERE / "config.json", encoding="utf-8"))
    g = data["geometry"]
    g["n_x"], g["n_y"] = s["n_x"], s["n_y"]
    g["n_t"], g["end_time"] = s["n_t"], s["end_time"]
    data["delta"], data["delta_flow"] = s["eps_T"], s["eps_flow"]
    data["epsilon"] = 1.0 / math.sqrt(s["penalty_C"])
    cfg = ExperimentConfig.model_validate(data)
    z = np.load(d / f"checkpoint_{int(round(time / s['dt']))}.npz", allow_pickle=True)
    vx = np.empty_like(z["sf"])
    vy = np.empty_like(z["sf"])
    calculate_velocity_from_sf(z["sf"], vx, vy, cfg)
    sp = np.hypot(vx, vy) * cfg.v
    liquid = z["u"] > 0
    return float(sp[liquid].max()) if liquid.any() else float("nan")


def main(argv=None):
    a = parse_args(argv)
    runs = load(a.family)

    As = [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
    av, vs = [], []
    for A in As:
        k = (151, 0.02, 0.1, A)
        if k not in runs:
            continue
        av.append(A)
        vs.append(runs[k][0]["max_speed_in_solid"])

    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)

    ax.loglog(av, vs, "o-", ms=4, color="tab:blue", label=r"largest $|v|$ in the ice")
    ref = np.array([av[0], av[-1]], dtype=float)
    ax.loglog(ref, vs[0] * ref[0] / ref, ":", color="0.6", lw=1.0)
    ax.text(ref[1] * 0.55, vs[0] * ref[0] / ref[1] * 1.7, r"$A^{-1}$",
            fontsize=8.5, color="0.45")

    s0, d0 = runs[(151, 0.02, 0.1, 10 ** 6)]
    vl = liquid_speed(s0, d0, a.time)
    ax.axhline(vl, color="tab:red", ls="--", lw=1.2,
               label=r"largest $|v|$ in the water")

    ax.set(xlabel=r"$A$, s$^{-1}$", ylabel=r"$|v|$, m s$^{-1}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="lower left")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    print("->", a.out)
    print(f"   largest |v| in the water at A = 1e6: {vl * 1e3:.3f} mm/s")
    for A, v in zip(av, vs):
        print(f"   A = {A:.0e}: |v| in ice {v:.2e} m/s, ratio to water {v / vl:.1e}")


if __name__ == "__main__":
    main()
