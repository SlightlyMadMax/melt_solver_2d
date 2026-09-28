"""
Figures for the horizontal layer: interface history, final interface shape and the
energy balance. Takes any number of run directories written by horizontal_layer/run.py.

    python plot_layer_runs.py --runs DIR [DIR ...] --labels "..." --outdir DIR
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.parameters.config import ExperimentConfig  # noqa: E402
from src.examples.horizontal_layer.run import (  # noqa: E402
    interface_heights, liquid_fraction, mean_interface_y,
)
from src.utils.nusselt import calculate_nusselt  # noqa: E402

LAYER = ROOT / "src/examples/horizontal_layer"

mpl.rcParams.update({
    "font.size": 12, "axes.labelsize": 11, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "lines.linewidth": 1.4,
    "figure.dpi": 300,
})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--old", action=argparse.BooleanOptionalAction, default=True,
                   help="also draw the stored runs of the published scheme")
    p.add_argument("--outdir", type=Path, default=LAYER / "graphs" / "new_scheme")
    return p.parse_args(argv)


def checkpoints(d: Path):
    fs = sorted(d.glob("checkpoint_*.npz"), key=lambda p: int(re.search(r"_(\d+)", p.name).group(1)))
    return [(int(re.search(r"_(\d+)", f.name).group(1)), f) for f in fs]


def history(d: Path, dt: float, cfg):
    t, y, frac, nu = [], [], [], []
    for n, f in checkpoints(d):
        u = np.load(f, allow_pickle=True)["u"]
        t.append(n * dt / 3600.0)
        y.append(mean_interface_y(u, cfg) * 1e3)
        frac.append(liquid_fraction(u, cfg))
        nu.append(calculate_nusselt(u=u, cfg=cfg, wall="top"))
    return np.array(t), np.array(y), np.array(frac), np.array(nu)


def main(argv=None):
    a = parse_args(argv)
    cfg = ExperimentConfig.load_from_file(str(LAYER / "config.json"))
    labels = a.labels or [d.name for d in a.runs]
    a.outdir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(a.runs)))

    fig_h, ax_h = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    fig_p, ax_p = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    fig_d, (ax_f, ax_n) = plt.subplots(2, 1, figsize=(5.4, 5.0), sharex=True,
                                       constrained_layout=True)

    for (d, label, c) in zip(a.runs, labels, colors):
        s = json.load(open(d / "summary.json", encoding="utf-8"))
        t, y, frac, nu = history(d, s["dt"], cfg)
        ax_h.plot(t, y, color=c, label=label)
        ax_f.plot(t, frac, color=c, label=label)
        ax_n.plot(t, nu, color=c, label=label)
        u = np.load(checkpoints(d)[-1][1], allow_pickle=True)["u"]
        x = np.arange(cfg.geometry.n_x) * cfg.geometry.dx * 1e3
        ax_p.plot(x, interface_heights(u, cfg) * 1e3, color=c, label=label)

    if a.old:
        for case, style in (("melting", "--"), ("freezing", ":")):
            fs = checkpoints(LAYER / "data" / case)
            if not fs:
                continue
            t = np.array([n * 0.1 / 3600 for n, _ in fs[:: max(1, len(fs) // 120)]])
            y = np.array([mean_interface_y(np.load(f)["u"], cfg) * 1e3
                          for _, f in fs[:: max(1, len(fs) // 120)]])
            ax_h.plot(t, y, style, color="0.45", lw=1.1, label=f"published scheme, {case}")
            u = np.load(fs[-1][1])["u"]
            x = np.arange(cfg.geometry.n_x) * cfg.geometry.dx * 1e3
            ax_p.plot(x, interface_heights(u, cfg) * 1e3, style, color="0.45", lw=1.1,
                      label=f"published scheme, {case}")

    ax_h.set(xlabel=r"$t$, h", ylabel="mean interface height, mm")
    ax_h.grid(alpha=0.25)
    ax_h.legend(frameon=False)
    fig_h.savefig(a.outdir / "interface_vs_time.png", dpi=300, bbox_inches="tight")

    ax_f.set(ylabel="liquid fraction")
    ax_f.grid(alpha=0.25)
    ax_f.legend(frameon=False)
    ax_n.set(xlabel=r"$t$, h", ylabel=r"$\overline{Nu}$ on the top wall")
    ax_n.grid(alpha=0.25)
    fig_d.savefig(a.outdir / "fraction_and_nusselt.png", dpi=300, bbox_inches="tight")

    ax_p.set(xlabel=r"$x$, mm", ylabel="interface height, mm")
    ax_p.grid(alpha=0.25)
    ax_p.legend(frameon=False, ncol=2)
    fig_p.savefig(a.outdir / "interface_profile_final.png", dpi=300, bbox_inches="tight")

    # energy balance, one panel per run
    fig_e, axes = plt.subplots(len(a.runs), 1, figsize=(5.4, 2.6 * len(a.runs)),
                               sharex=True, constrained_layout=True, squeeze=False)
    for ax, d, label in zip(axes[:, 0], a.runs, labels):
        f = d / "energy_budget.npz"
        if not f.exists():
            ax.text(0.5, 0.5, "no energy budget", ha="center", transform=ax.transAxes)
            continue
        e = dict(np.load(f))
        walls = [k for k in ("Q_left", "Q_right", "Q_bottom", "Q_top") if k in e]
        if not walls:  # files written before the ledger recorded each wall
            walls = ["Q_hot", "Q_cold", "Q_adiabatic"]
        q = sum(e[k] for k in walls)
        t_h = e["t"] / 3600
        dE = e["E_sens"] + e["E_lat"] - e["E0_sens"] - e["E0_lat"]
        wall = np.cumsum(q * e["dt"])
        res = dE - wall
        ax.plot(t_h, dE / 1e3, color="k", label=r"$\Delta E$")
        ax.plot(t_h, wall / 1e3, "--", color="0.55", label=r"$\int_0^t Q\,dt$")
        ax.plot(t_h, np.cumsum(e["defect_advection"]) / 1e3, color="tab:red", lw=3.0,
                alpha=0.35, solid_capstyle="round", label="convection defect")
        ax.plot(t_h, res / 1e3, color="tab:purple", lw=1.0, label="residual")
        ax.set_ylabel("energy, kJ m$^{-1}$")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="center right")
        # Yardstick: the heat actually exchanged with the driving walls, as recorded
        # by the ledger. The NET wall heat tends to zero at steady state and would
        # make any ratio meaningless. The residual goes in the title, where it cannot
        # collide with the curves.
        j = d / "energy_budget.json"
        note = f"residual {res[-1] / 1e3:+.1f} kJ m$^{{-1}}$"
        if j.exists():
            thr = json.load(open(j, encoding="utf-8"))["throughput_J_per_m"]
            note += f", {100 * res[-1] / thr:+.2f}% of the heat exchanged with the walls"
        ax.set_title(f"{label} — {note}", fontsize=9)
    axes[-1, 0].set_xlabel(r"$t$, h")
    fig_e.savefig(a.outdir / "energy_balance.png", dpi=300, bbox_inches="tight")
    print("->", a.outdir)


if __name__ == "__main__":
    main()
