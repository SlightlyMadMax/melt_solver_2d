#!/usr/bin/env python
"""
Phase interface of the square cavity under each numerical parameter, at t = 2340 s.

    a  grid spacing        b  time step        c  smoothing width
    d  largest speed left in the ice against the penalty coefficient

    python -m src.examples.water_freezing.plot_param_interfaces
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
                   default=HERE / "graphs" / "new_scheme" / "parameter_interfaces.png")
    return p.parse_args(argv)


def front(u):
    """Position of the first T = Tm crossing in each row, as a fraction of the width."""
    n_y, n_x = u.shape
    cross = u[:, :-1] * u[:, 1:] < 0
    xs = np.full(n_y, np.nan)
    for j in range(n_y):
        idx = np.nonzero(cross[j])[0]
        if idx.size:
            i = idx[0]
            a, b = u[j, i], u[j, i + 1]
            xs[j] = (i + a / (a - b)) / (n_x - 1)
    return xs


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


def draw(ax, runs, keys, labels, colors):
    for k, lab, c in zip(keys, labels, colors):
        s, d = runs[k]
        f = d / f"checkpoint_{int(round(2340.0 / s['dt']))}.npz"
        if not f.exists():
            continue
        xs = front(np.load(f, allow_pickle=True)["u"])
        y = np.linspace(0.0, 1.0, s["n_y"])
        good = np.isfinite(xs)
        ax.plot(xs[good], y[good], color=c, label=lab)
    ax.set(xlim=(0.58, 0.74), ylim=(0.0, 1.0))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")


def liquid_speed(s, d: Path) -> float:
    """Largest speed in the water of that run, from the stored stream function."""
    data = json.load(open(HERE / "config.json", encoding="utf-8"))
    g = data["geometry"]
    g["n_x"], g["n_y"] = s["n_x"], s["n_y"]
    g["n_t"], g["end_time"] = s["n_t"], s["end_time"]
    data["delta"], data["delta_flow"] = s["eps_T"], s["eps_flow"]
    data["epsilon"] = 1.0 / math.sqrt(s["penalty_C"])
    cfg = ExperimentConfig.model_validate(data)
    z = np.load(d / f"checkpoint_{int(round(2340.0 / s['dt']))}.npz", allow_pickle=True)
    vx = np.empty_like(z["sf"]); vy = np.empty_like(z["sf"])
    calculate_velocity_from_sf(z["sf"], vx, vy, cfg)
    sp = np.hypot(vx, vy) * cfg.v
    liquid = z["u"] > 0
    return float(sp[liquid].max()) if liquid.any() else float("nan")


def main(argv=None):
    a = parse_args(argv)
    runs = load(a.family)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, 4))

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.4), constrained_layout=True)

    grids = [51, 101, 151, 201]
    draw(axes[0, 0], runs, [(n, 0.02, 0.1, 10 ** 6) for n in grids],
         [f"${n}\\times{n}$" for n in grids], cmap)
    axes[0, 0].set_ylabel(r"$Y$")

    dts = [0.01, 0.02, 0.05, 0.1]
    draw(axes[0, 1], runs, [(151, t, 0.1, 10 ** 6) for t in dts],
         [rf"$\tau = {t:g}$ s" for t in dts], cmap)

    eps = [0.05, 0.1, 0.2, 0.4]
    draw(axes[1, 0], runs, [(151, 0.02, e, 10 ** 6) for e in eps],
         [rf"$\varepsilon = {e:g}$ $^\circ$C" for e in eps], cmap)
    axes[1, 0].set(xlabel=r"$X$", ylabel=r"$Y$")
    axes[0, 1].set(xlabel="")
    axes[1, 1].set_xlabel(r"$A$, s$^{-1}$")

    As = [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
    vs, av = [], []
    for A in As:
        k = (151, 0.02, 0.1, A)
        if k not in runs:
            continue
        s, d = runs[k]
        av.append(A); vs.append(s["max_speed_in_solid"])
    ax = axes[1, 1]
    ax.loglog(av, vs, "o-", ms=4, color="tab:blue", label=r"largest $|v|$ in the ice")
    ref = np.array([av[0], av[-1]], dtype=float)
    ax.loglog(ref, vs[0] * ref[0] / ref, ":", color="0.6", lw=1.0)
    ax.text(ref[1] * 0.6, vs[0] * ref[0] / ref[1] * 1.6, r"$A^{-1}$",
            fontsize=8.5, color="0.45")
    s0, d0 = runs[(151, 0.02, 0.1, 10 ** 6)]
    vl = liquid_speed(s0, d0)
    ax.axhline(vl, color="tab:red", ls="--", lw=1.2,
               label=r"largest $|v|$ in the water")
    ax.set_ylabel(r"$|v|$, m s$^{-1}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="lower left")

    for ax, tag in zip(axes.ravel(), "abcd"):
        ax.text(0.5, 1.02, tag, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    print("->", a.out)
    print(f"   largest |v| in the water at A = 1e6: {vl * 1e3:.3f} mm/s")
    for A, v in zip(av, vs):
        print(f"   A = {A:.0e}: |v| in ice {v:.2e} m/s, ratio to water {v / vl:.1e}")


if __name__ == "__main__":
    main()
