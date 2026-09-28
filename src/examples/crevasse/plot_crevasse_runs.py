"""
Figures for the crevasse: temperature collage, flow, ice fraction, water depth on
the crevasse axis and the energy balance. Takes run directories written by
crevasse/run.py.

    python plot_crevasse_runs.py --run DIR [--conduction DIR] [--times 0.25 2 8 24]
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.core.constants import ABS_ZERO  # noqa: E402
from src.parameters.config import ExperimentConfig  # noqa: E402

CREV = ROOT / "src/examples/crevasse"

mpl.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
    }
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--run", type=Path,
                   default=CREV / "data/new_scheme/convection_201_dt0.2")
    p.add_argument("--conduction", type=Path, default=None,
                   help="optional conduction-only counterpart (a --no-flow run)")
    p.add_argument("--times", type=float, nargs="+", default=[0.25, 2.0, 8.0, 24.0],
                   help="hours shown in the collage and the flow figure")
    p.add_argument("--outdir", type=Path, default=CREV / "graphs" / "new_scheme")
    return p.parse_args(argv)


def config_for(d: Path) -> ExperimentConfig:
    """Rebuild the configuration the run was started with, from its summary."""
    s = json.load(open(d / "summary.json", encoding="utf-8"))
    data = json.load(open(CREV / "convection" / "config.json", encoding="utf-8"))
    g = data["geometry"]
    g["n_x"], g["n_y"] = s["n_x"], s["n_y"]
    g["n_t"], g["end_time"] = s["n_t"], s["end_time"]
    data["delta"], data["delta_flow"] = s["eps_T"], s["eps_flow"]
    data["epsilon"] = 1.0 / math.sqrt(s["penalty_C"])
    return ExperimentConfig.model_validate(data)


def checkpoints(d: Path):
    """(absolute time [s], path) for every checkpoint, oldest first."""
    s = json.load(open(d / "summary.json", encoding="utf-8"))
    offset = s.get("time_offset", 0.0)
    out = []
    for f in d.glob("checkpoint_*.npz"):
        n = int(re.search(r"_(\d+)", f.name).group(1))
        out.append((n * s["dt"] + offset, f))
    return sorted(out)


def nearest(cps, t_hours):
    return min(cps, key=lambda p: abs(p[0] - t_hours * 3600.0))


def temperature(f: Path, cfg) -> np.ndarray:
    u = np.load(f, allow_pickle=True)["u"]
    return u * cfg.delta_u + cfg.u_ref + ABS_ZERO


def time_label(t_s: float) -> str:
    # Checkpoints fall on a 10 min grid offset by the 60 s first stage, so a request
    # for "2 h" lands a minute off; round the label rather than print 2.01667.
    if t_s >= 3600:
        return rf"$t = {round(t_s / 3600, 1):g}$ h"
    if t_s >= 60:
        return rf"$t = {int(round(t_s / 60))}$ min"
    return rf"$t = {int(round(t_s))}$ s"


def collage(a, rows, out: Path):
    """One row per case, one column per requested instant."""
    n_row, n_col = len(rows), len(a.times)
    fields = []
    for _, d, cfg in rows:
        cps = checkpoints(d)
        fields.append([(t, temperature(f, cfg))
                       for t, f in (nearest(cps, h) for h in a.times)])
    vmin = min(arr.min() for row in fields for _, arr in row)
    vmax = max(arr.max() for row in fields for _, arr in row)

    # The panels are square, so the height has to follow the column width, or
    # constrained_layout spreads the rows apart with white space.
    fig, axes = plt.subplots(n_row, n_col, figsize=(6.3, 1.5 * n_row + 0.5),
                             constrained_layout=True, squeeze=False)
    im = None
    for i, ((label, d, cfg), row) in enumerate(zip(rows, fields)):
        X, Y = cfg.geometry.mesh_grid
        for j, (t_s, arr) in enumerate(row):
            ax = axes[i, j]
            im = ax.pcolormesh(X, Y, arr, cmap="Blues", vmin=vmin, vmax=vmax,
                               shading="auto")
            ax.contour(X, Y, arr, levels=[0.0], colors="black", linewidths=0.8)
            ax.set_box_aspect(1)
            ax.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.1, 0.2]))
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.1, 0.2]))
            ax.tick_params(labelleft=(j == 0), labelbottom=(i == n_row - 1))
            if j == 0:
                ax.set_ylabel(r"$y$, m")
            if i == n_row - 1:
                ax.set_xlabel(r"$x$, m")
            ax.text(0.98, 0.025, time_label(t_s), transform=ax.transAxes,
                    fontsize=8, va="bottom", ha="right")
        if n_row > 1:
            axes[i, 0].text(-0.62, 0.5, "ab"[i], transform=axes[i, 0].transAxes,
                            fontweight="bold", va="center", ha="right", clip_on=False)
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02,
                        ticks=np.linspace(vmin, vmax, 6))
    cbar.set_label("Temperature, °C")
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")


def flow(a, d: Path, cfg, out: Path):
    """Streamlines in the water, with the 0 degC and 4 degC isotherms."""
    cps = checkpoints(d)
    X, Y = cfg.geometry.mesh_grid
    x, y = X[0, :], Y[:, 0]
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("0.85")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0), constrained_layout=True)
    im = None
    for ax, h in zip(axes.ravel(), a.times):
        t_s, f = nearest(cps, h)
        z = np.load(f, allow_pickle=True)
        t_c = z["u"] * cfg.delta_u + cfg.u_ref + ABS_ZERO
        v_x, v_y = z["v_x"] * cfg.v, z["v_y"] * cfg.v
        liquid = t_c >= 0.0
        water = np.ma.masked_where(~liquid, t_c)
        im = ax.pcolormesh(X, Y, water, cmap=cmap, vmin=0.0, vmax=5.0, shading="auto")
        speed = np.hypot(v_x, v_y)
        v_max = speed[liquid].max() if liquid.any() else 0.0
        if v_max > 0:
            ax.streamplot(x, y, np.ma.masked_where(~liquid, v_x),
                          np.ma.masked_where(~liquid, v_y), color="black",
                          linewidth=0.3 + 1.8 * np.ma.masked_where(~liquid, speed) / v_max,
                          density=2.0, arrowsize=0.6)
        ax.contour(X, Y, t_c, levels=[0.0], colors="black", linewidths=1.2)
        if float(water.max()) > 4.0:
            ax.contour(X, Y, water, levels=[4.0], colors="white", linewidths=1.0,
                       linestyles="--")
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.05, 0.10, 0.15, 0.20]))
        ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.05, 0.10, 0.15, 0.20]))
        ax.set_xlim(0.0, cfg.geometry.width)
        ax.set_ylim(0.0, cfg.geometry.height)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$, m")
        ax.set_ylabel(r"$y$, m")
        ax.set_title(rf"{time_label(t_s)}, $|\mathbf{{v}}|_\mathrm{{max}} = "
                     rf"{v_max * 1e3:.2f}$ mm/s", fontsize=10, pad=5)
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.6, pad=0.02,
                        ticks=np.linspace(0, 5, 6))
    cbar.set_label("Water temperature, °C")
    fig.savefig(out, dpi=300, bbox_inches="tight")


def histories(rows):
    """Ice fraction and the depth of the water column on the crevasse axis."""
    out = []
    for label, d, cfg in rows:
        t, ice, depth = [], [], []
        for t_s, f in checkpoints(d):
            u = np.load(f, allow_pickle=True)["u"]
            t.append(t_s / 3600.0)
            ice.append(float(np.mean(u < cfg.u_pt_nd)))
            column = u[:, u.shape[1] // 2] > cfg.u_pt_nd
            k = 0
            for liquid in reversed(column):
                if not liquid:
                    break
                k += 1
            depth.append(k * cfg.geometry.dy)
        out.append((label, np.array(t), np.array(ice), np.array(depth)))
    return out


def energy_panel(d: Path, out: Path, label: str) -> bool:
    f = d / "energy_budget.npz"
    if not f.exists():
        return False
    e = dict(np.load(f))
    walls = [k for k in ("Q_left", "Q_right", "Q_bottom", "Q_top") if k in e]
    q = sum(e[k] for k in walls)
    t_h = e["t"] / 3600
    dE = e["E_sens"] + e["E_lat"] - e["E0_sens"] - e["E0_lat"]
    wall = np.cumsum(q * e["dt"])
    res = dE - wall
    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    ax.plot(t_h, dE / 1e3, color="k", label=r"$\Delta E$")
    ax.plot(t_h, wall / 1e3, "--", color="0.55", label=r"$\int_0^t Q\,dt$")
    ax.plot(t_h, np.cumsum(e["defect_advection"]) / 1e3, color="tab:red", lw=3.0,
            alpha=0.35, solid_capstyle="round", label="convection defect")
    ax.plot(t_h, res / 1e3, color="tab:purple", lw=1.0, label="residual")
    note = f"residual {res[-1] / 1e3:+.1f} kJ m$^{{-1}}$"
    j = d / "energy_budget.json"
    if j.exists():
        thr = json.load(open(j, encoding="utf-8"))["throughput_J_per_m"]
        note += f", {100 * res[-1] / thr:+.2f}% of the heat exchanged with the walls"
    ax.set_title(f"{label} — {note}", fontsize=9)
    ax.set(xlabel=r"$t$, h", ylabel="energy, kJ m$^{-1}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    return True


def main(argv=None):
    a = parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)
    rows = [("convection", a.run, config_for(a.run))]
    if a.conduction is not None:
        rows.insert(0, ("conduction", a.conduction, config_for(a.conduction)))

    collage(a, rows, a.outdir / "collage.tiff")
    flow(a, a.run, rows[-1][2], a.outdir / "flow.png")

    hist = histories(rows)
    colors = {"convection": "tab:blue", "conduction": "tab:orange"}
    fig, ax = plt.subplots(figsize=(6.3, 3.6), constrained_layout=True)
    for label, t, ice, _ in hist:
        ax.plot(t, ice, color=colors[label], label=label)
    ax.set(xlabel="Time, h", ylabel="Ice fraction")
    ax.grid(alpha=0.25)
    if len(hist) > 1:
        ax.legend(frameon=False)
    fig.savefig(a.outdir / "ice_fraction.tiff", dpi=300, bbox_inches="tight")
    fig.savefig(a.outdir / "ice_fraction.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6.3, 3.6), constrained_layout=True)
    for label, t, _, depth in hist:
        ax.plot(t, depth * 1e3, color=colors[label], label=label)
    ax.set(xlabel="Time, h", ylabel="Water depth on the crevasse axis, mm")
    ax.grid(alpha=0.25)
    if len(hist) > 1:
        ax.legend(frameon=False)
    fig.savefig(a.outdir / "water_depth.png", dpi=300, bbox_inches="tight")

    energy_panel(a.run, a.outdir / "energy_balance.png", "crevasse, new scheme")

    for label, t, ice, depth in hist:
        print(f"{label}: ice {ice[0]:.4f} -> {ice[-1]:.4f}, "
              f"water depth {depth[0] * 1e3:.1f} -> {depth[-1] * 1e3:.1f} mm, "
              f"over {t[-1]:.2f} h")
    print("->", a.outdir)


if __name__ == "__main__":
    main()
