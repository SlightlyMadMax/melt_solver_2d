#!/usr/bin/env python
"""
Domain-integrated energy budget of one run, as a function of time.

Reads the energy_budget.npz written by run.py --energy-budget and plots

  (a) heat through the hot and cold walls,
  (b) change of sensible, latent and total energy since t = 0 against the net
      wall heat,
  (c) the residual  dE - int Q dt  against a +-1 % band of |dE|.

Time is logarithmic so the start-up, where the phase change is fastest, is
readable next to the rest of the run. All quantities are per unit depth.

Example
-------
    python -m src.examples.water_freezing.plot_energy_budget \
        --run-dir src/examples/water_freezing/data/warm_start/energy_budget_implicit/151x151_dt0.02_e0.1_ef0.1_C1e+06 \
        --out src/examples/water_freezing/graphs/warm_start/energy_budget.png
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.5,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "lines.linewidth": 1.3,
        "figure.dpi": 300,
    }
)

SCHEME_NAMES = {"cn": "Crank-Nicolson", "dr": "Douglas-Rachford", "implicit": "implicit"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--run-dir", type=Path, required=True,
                   help="run directory holding energy_budget.npz and summary.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--points", type=int, default=1500,
                   help="log-spaced samples kept for drawing")
    return p.parse_args(argv)


def log_sample(n: int, points: int, head: int = 200) -> np.ndarray:
    """Indices dense at the start, where the steps matter, log-spaced afterwards."""
    tail = np.unique(np.geomspace(1, n, points).astype(int) - 1)
    return np.unique(np.concatenate([np.arange(min(head, n)), tail]))


def title_of(s: dict) -> tuple[str, str]:
    scheme = SCHEME_NAMES.get(s.get("penalty_time_scheme", "cn"), s.get("penalty_time_scheme"))
    head = f"{s['start']} start, {scheme} penalty"
    eps = (rf"$\varepsilon = {s['eps_T']:g}$ K" if s["eps_T"] == s["eps_flow"]
           else rf"$\varepsilon_T = {s['eps_T']:g}$ K, "
                rf"$\varepsilon_\mathrm{{flow}} = {s['eps_flow']:g}$ K")
    sub = (rf"${s['n_x']}\times{s['n_y']}$, $\tau = {s['dt']:g}$ s, {eps}, "
           rf"$C = 10^{{{int(round(np.log10(s['penalty_C'])))}}}$ s$^{{-1}}$")
    return head, sub


def main(argv=None) -> None:
    args = parse_args(argv)
    a = dict(np.load(args.run_dir / "energy_budget.npz"))
    with open(args.run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    t, dt = a["t"], a["dt"]
    wall = a["Q_hot"] + a["Q_cold"] + a["Q_adiabatic"]
    d_sens = a["E_sens"] - a["E0_sens"]
    d_lat = a["E_lat"] - a["E0_lat"]
    d_total = d_sens + d_lat
    wall_heat = np.cumsum(wall * dt)
    residual = d_total - wall_heat  # = cumulative capacity + advection defect

    i = log_sample(t.size, args.points)
    kj = 1e-3

    fig, axes = plt.subplots(3, 1, figsize=(3.6, 6.4), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.0, 1.15, 1.0]})
    ax_q, ax_e, ax_r = axes

    # (a) wall heat. Both drawn as magnitudes so one log axis holds them
    ax_q.plot(t[i], a["Q_hot"][i], color="tab:red", label=r"$Q_\mathrm{hot}$ (in)")
    ax_q.plot(t[i], -a["Q_cold"][i], color="tab:blue", label=r"$-Q_\mathrm{cold}$ (out)")
    ax_q.set_yscale("log")
    ax_q.set_ylabel(r"boundary heat flow, W m$^{-1}$")
    ax_q.legend(loc="upper right", frameon=False)

    # (b) energy change since t = 0 and the heat that crossed the walls
    ax_e.plot(t[i], kj * d_sens[i], color="tab:orange", label=r"$\Delta E_\mathrm{sens}$")
    ax_e.plot(t[i], kj * d_lat[i], color="tab:green", label=r"$\Delta E_\mathrm{lat}$")
    ax_e.plot(t[i], kj * d_total[i], color="k", label=r"$\Delta E$")
    ax_e.plot(t[i], kj * wall_heat[i], color="0.55", ls="--",
              label=r"$\int_0^t (Q_\mathrm{hot}+Q_\mathrm{cold})\,dt$")
    ax_e.axhline(0.0, color="0.8", lw=0.6, zorder=0)
    ax_e.set_ylabel(r"energy change, kJ m$^{-1}$")
    ax_e.legend(loc="lower left", frameon=False)

    # (c) residual, with a band of 1 % of the energy change for scale
    band = 0.01 * np.abs(d_total)
    ax_r.fill_between(t[i], -kj * band[i], kj * band[i], color="0.9", lw=0,
                      label=r"$\pm 1\%$ of $|\Delta E|$")
    ax_r.plot(t[i], kj * residual[i], color="tab:purple",
              label=r"$\Delta E - \int_0^t Q\,dt$")
    ax_r.axhline(0.0, color="0.7", lw=0.6)
    ax_r.set_ylabel(r"residual, kJ m$^{-1}$")
    ax_r.set_xlabel(r"$t$, s")
    ax_r.set_xscale("log")
    ax_r.set_xlim(t[0], t[-1])
    ax_r.legend(loc="upper center", frameon=False)

    # The final value goes where the curve is not: a positive residual ends high on
    # the right, so bottom-right is free; a negative one runs along the bottom, which
    # leaves the gap between the band and the curve on the left
    rel = residual[-1] / abs(d_total[-1])
    if residual[-1] > 0:
        xy, ha, va = (0.98, 0.05), "right", "bottom"
    else:
        xy, ha, va = (0.03, 0.45), "left", "center"
    note = f"{100 * rel:+.2f}".replace("-", "−")
    ax_r.annotate(rf"{note}% at $t = {t[-1]:.0f}$ s", xy=xy, xycoords="axes fraction",
                  ha=ha, va=va, fontsize=8.5, color="tab:purple")

    # Panel letters sit outside the frame: inside, every corner is taken by a
    # legend or by data in one of the two start types
    for ax, tag in zip(axes, "abc"):
        ax.text(-0.02, 1.0, f"({tag})", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10)
        ax.grid(alpha=0.25, which="major")

    head, sub = title_of(summary)
    ax_q.set_title(head, fontsize=9, pad=13)
    ax_q.text(0.5, 1.012, sub, transform=ax_q.transAxes, ha="center", va="bottom",
              fontsize=7.5, color="0.35")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"-> {args.out}")
    print(f"   dE = {d_total[-1]:+.4e} J/m, int Q dt = {wall_heat[-1]:+.4e} J/m, "
          f"residual = {residual[-1]:+.4e} J/m ({100 * rel:+.3f}% of |dE|)")


if __name__ == "__main__":
    main()
