#!/usr/bin/env python
"""
Phase interface of the square cavity under each numerical parameter, at t = 2340 s.

    a  grid spacing      b  time step
    c  smoothing width   d  penalty coefficient

    python -m src.examples.water_freezing.plot_param_interfaces
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
    p.add_argument("--family", type=Path, default=DATA / "warm_start" / "star_sigma05")
    p.add_argument("--out", type=Path,
                   default=HERE / "graphs" / "new_scheme" / "parameter_interfaces.png")
    p.add_argument("--time", type=float, default=2340.0)
    p.add_argument("--xlim", type=float, nargs=2, default=(0.585, 0.74))
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


def draw(ax, runs, keys, labels, colors, time):
    for k, lab, c in zip(keys, labels, colors):
        if k not in runs:
            continue
        s, d = runs[k]
        f = d / f"checkpoint_{int(round(time / s['dt']))}.npz"
        if not f.exists():
            continue
        xs = front(np.load(f, allow_pickle=True)["u"])
        y = np.linspace(0.0, 1.0, s["n_y"])
        good = np.isfinite(xs)
        ax.plot(xs[good], y[good], color=c, label=lab)
    ax.set(ylim=(0.0, 1.0))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")


def main(argv=None):
    a = parse_args(argv)
    runs = load(a.family)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, 4))

    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.4), sharex=True, sharey=True,
                             constrained_layout=True)

    grids = [51, 101, 151, 201]
    draw(axes[0, 0], runs, [(n, 0.02, 0.1, 10 ** 6) for n in grids],
         [f"${n}\\times{n}$" for n in grids], cmap, a.time)

    dts = [0.01, 0.02, 0.05, 0.1]
    draw(axes[0, 1], runs, [(151, t, 0.1, 10 ** 6) for t in dts],
         [rf"$\tau = {t:g}$ s" for t in dts], cmap, a.time)

    eps = [0.05, 0.1, 0.2, 0.4]
    draw(axes[1, 0], runs, [(151, 0.02, e, 10 ** 6) for e in eps],
         [rf"$\varepsilon = {e:g}$ $^\circ$C" for e in eps], cmap, a.time)

    As = [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
    draw(axes[1, 1], runs, [(151, 0.02, 0.1, A) for A in As],
         [rf"$A = 10^{{{int(round(np.log10(A)))}}}$ s$^{{-1}}$" for A in As],
         cmap, a.time)

    axes[0, 0].set_ylabel(r"$Y$")
    axes[1, 0].set(xlabel=r"$X$", ylabel=r"$Y$", xlim=tuple(a.xlim))
    axes[1, 1].set_xlabel(r"$X$")

    for ax, tag in zip(axes.ravel(), "abcd"):
        ax.text(0.5, 1.02, tag, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    print("->", a.out)
    for lab, keys in (("grid", [(n, 0.02, 0.1, 10 ** 6) for n in grids]),
                      ("dt", [(151, t, 0.1, 10 ** 6) for t in dts]),
                      ("eps", [(151, 0.02, e, 10 ** 6) for e in eps]),
                      ("A", [(151, 0.02, 0.1, A) for A in As])):
        xs = [runs[k][0]["mean_interface_x"] for k in keys if k in runs]
        print(f"   {lab:5} mean X over the set: "
              + ", ".join(f"{v:.4f}" for v in xs)
              + f"   spread {100 * (max(xs) - min(xs)) / np.mean(xs):.3f} %")


if __name__ == "__main__":
    main()
