#!/usr/bin/env python
"""
Convergence of the square-cavity solidification with the numerical parameters.

    a  deviation of the interface position, the ice fraction and the Nusselt number
       from the solution on the finest grid, against the grid spacing, beside the
       residual of the energy balance.
    b  residual of the energy balance against the time step and against the smoothing
       width, each relative to the value adopted in Section 4.2.

    python -m src.examples.water_freezing.plot_sensitivity
"""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
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
    p.add_argument("--family", type=Path,
                   default=DATA / "warm_start" / "star_sigma05")
    p.add_argument("--out", type=Path,
                   default=HERE / "graphs" / "new_scheme" / "sensitivity.png")
    return p.parse_args(argv)


def load(family: Path):
    runs = {}
    for d in sorted(glob.glob(str(family / "*"))):
        sj, ej = os.path.join(d, "summary.json"), os.path.join(d, "energy_budget.json")
        if not (os.path.exists(sj) and os.path.exists(ej)):
            continue
        s = json.load(open(sj, encoding="utf-8"))
        e = json.load(open(ej, encoding="utf-8"))
        key = (s["n_x"], round(s["dt"], 4), round(s["eps_T"], 4), round(s["penalty_C"]))
        runs.setdefault(key, (s, e))
    return runs


def main(argv=None):
    a = parse_args(argv)
    runs = load(a.family)
    pct = 100.0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.3, 3.0), constrained_layout=True)

    # --- a: grid refinement -------------------------------------------------
    grids = [21, 31, 51, 101, 151, 201]
    sel = [(n, 0.02, 0.1, 10 ** 6) for n in grids]
    h = np.array([38.0 / (n - 1) for n in grids])          # mm
    ref = runs[sel[-1]][0]
    series = {
        r"$X$ of the front": [runs[k][0]["mean_interface_x"] for k in sel],
        "ice fraction": [runs[k][0]["ice_fraction"] for k in sel],
        r"$\overline{Nu}$": [runs[k][0]["Nu_hot"] for k in sel],
    }
    keys = ["mean_interface_x", "ice_fraction", "Nu_hot"]
    markers = ["o", "s", "^"]
    for (label, vals), key, m in zip(series.items(), keys, markers):
        d = pct * np.abs(np.array(vals) - ref[key]) / abs(ref[key])
        ax0.loglog(h[:-1], d[:-1], m + "-", ms=4, label=label)
    res = pct * np.abs([runs[k][1]["defect_total_rel_throughput"] for k in sel])
    ax0.loglog(h, res, "d--", ms=4, color="0.4", label="energy residual")
    g = np.array([h[0], h[-2]])
    ax0.loglog(g, 2.0 * (g / g[0]) ** 2, ":", color="0.6", lw=1.0)
    ax0.text(g[0] * 0.95, 2.4, r"$h^{2}$", fontsize=8.5, color="0.45", ha="left")
    ax0.set(xlabel=r"grid spacing $h$, mm", ylabel="deviation and residual, %")
    ax0.grid(alpha=0.25, which="both")
    ax0.legend(frameon=False, loc="lower right")

    # --- b: time step and smoothing width -----------------------------------
    dts = [0.01, 0.02, 0.05, 0.1]
    eps = [0.05, 0.1, 0.2, 0.4]
    r_dt = pct * np.abs([runs[(151, t, 0.1, 10 ** 6)][1]["defect_total_rel_throughput"]
                         for t in dts])
    r_ep = pct * np.abs([runs[(151, 0.02, e, 10 ** 6)][1]["defect_total_rel_throughput"]
                         for e in eps])
    ax1.loglog(np.array(dts) / 0.02, r_dt, "o-", ms=4, label=r"time step $\tau$")
    ax1.loglog(np.array(eps) / 0.1, r_ep, "s--", ms=4, label=r"smoothing width $\varepsilon$")
    ax1.set(xlabel="parameter relative to the adopted value",
            ylabel="energy residual, %")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(frameon=False, loc="upper left")

    for ax, tag in zip((ax0, ax1), "ab"):
        ax.text(0.5, 1.02, tag, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    print("->", a.out)
    for (label, vals), key in zip(series.items(), keys):
        d = pct * np.abs(np.array(vals) - ref[key]) / abs(ref[key])
        print(f"  {label:20} " + ", ".join(f"{n}:{x:.3f}%" for n, x in zip(grids, d)))


if __name__ == "__main__":
    main()
