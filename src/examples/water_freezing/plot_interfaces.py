#!/usr/bin/env python
"""
Overlay the phase interface at the comparison time for one parameter axis.

Companion to the star-design study: one figure per axis (grid, time step, smoothing
width, penalty coefficient), each showing the T = T_m contour for every value on that
axis so the convergence can be read off directly.

The interface is taken as the longest T = T_m contour path rather than the scattered
crossing points of `get_phase_trans_boundary`, which is unordered and plots as a point
cloud once several curves are overlaid.

Examples
--------
Every axis found in the star-sweep output:
    python -m src.examples.water_freezing.plot_interfaces

One axis only, shown on screen:
    python -m src.examples.water_freezing.plot_interfaces --axis grid --show

The warm-start runs from the earlier sweep:
    python -m src.examples.water_freezing.plot_interfaces \
        --data-dir data/warm_start/refinement_dt0.01

The grid axis splits into two figures, because the very coarse grids and the converged
family answer different questions and cannot share one pair of axes: the inset is only
readable while the curves it magnifies overlap. Note that a plain run of this script
draws all six grids into interface_grid.png, so rebuild both:
    python -m src.examples.water_freezing.plot_interfaces --axis grid \
        --grids 51 101 151 201 --deviation
    python -m src.examples.water_freezing.plot_interfaces --axis grid --no-inset \
        --name interface_grid_coarse --deviation
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PIV_CURVE = HERE / "data" / "inputs" / "piv_boundary.npz"

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
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

# Which summary field defines each axis, and how to label it
AXES = {
    "grid": ("n_x", lambda v: rf"${int(v)}\times{int(v)}$"),
    "dt": ("dt", lambda v: rf"$\tau = {v:g}$ s"),
    "eps": ("eps_T", lambda v: rf"$\varepsilon = {v:g}$ K"),
    "penalty": (
        "penalty_C",
        lambda v: rf"$C = 10^{{{int(round(np.log10(v)))}}}$ s$^{{-1}}$",
    ),
}

# Every parameter that must agree for two runs to sit on the same axis. eps_flow is
# listed even though no axis varies it alone: the study includes off-diagonal points
# where it differs from eps_T, and without this they would be pulled into the eps axis
# and plotted as duplicates of the diagonal ones.
MATCH_FIELDS = ("n_x", "dt", "eps_T", "eps_flow", "penalty_C")


def _others(axis: str) -> list[str]:
    """Fields that must match the baseline for a run to belong to `axis`."""
    field = AXES[axis][0]
    skip = {field, "eps_flow"} if axis == "eps" else {field}
    return [f for f in MATCH_FIELDS if f not in skip]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay phase interfaces for one parameter axis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=HERE / "data" / "cold_start" / "parametric",
        help="directory holding the run subdirectories",
    )
    p.add_argument(
        "--axis",
        choices=sorted(AXES) + ["eps_split", "auto"],
        default="auto",
        help="which axis to plot; 'auto' plots every axis that has 2+ values",
    )
    p.add_argument("--outdir", type=Path, default=HERE / "graphs" / "cold_start")
    p.add_argument("--show", action="store_true")
    p.add_argument(
        "--format",
        default="png",
        choices=["png", "tiff", "pdf", "svg"],
    )
    p.add_argument(
        "--no-inset",
        action="store_true",
        help="omit the magnified inset; converged curves overlap almost exactly, so "
        "without it the figure shows no detail",
    )
    p.add_argument(
        "--inset-half-height",
        type=float,
        default=0.004,
        help="half-height of the inset window [m]",
    )
    p.add_argument(
        "--inset-half-width",
        type=float,
        default=None,
        help="half-width of the inset window [m]. If omitted, the width is "
        "chosen automatically to tightly fit the curves in the band",
    )
    p.add_argument(
        "--deviation",
        action="store_true",
        help="also plot the pointwise interface deviation from the reference run of "
        "each axis. Converged curves differ by a fraction of a cell, which an overlay "
        "cannot resolve; this shows where along the interface the difference sits and "
        "how large it is in micrometres",
    )
    p.add_argument(
        "--grids",
        type=int,
        nargs="*",
        default=None,
        help="restrict the grid axis to these values of n_x. The inset only earns its "
        "place when the curves it magnifies overlap, so a run that is far off the "
        "others is better shown in a figure of its own",
    )
    p.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help="plot these run directories instead of an axis, in this order. Names are "
        "taken relative to --data-dir. Use it for comparisons that move more than one "
        "parameter at a time, which no single axis can express",
    )
    p.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="legend entries for --runs; by default they are built from whichever "
        "parameters actually differ between the chosen runs",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="basename of the output file, without extension; defaults to "
        "interface_<axis>",
    )
    p.add_argument(
        "--piv-curve",
        type=Path,
        nargs="?",
        const=PIV_CURVE,
        default=None,
        help="draw the measured front from the PIV frame alongside the computed "
        "ones. With no path it uses data/inputs/piv_boundary.npz, written by "
        "piv_boundary.py. Off by default: the experiment is a cubic cavity with "
        "conducting side walls and its own uncertainty, so it belongs on figures "
        "that are about agreement with it, not on ones about convergence",
    )
    p.add_argument(
        "--piv-background",
        type=Path,
        default=None,
        help="overlay the curves on the experimental PIV frame, e.g. "
        "data/inputs/kowalewski.png. Off by default: these are verification figures "
        "and the "
        "differences between converged curves are sub-millimetre, so a photographic "
        "background hides exactly what the figure is meant to show",
    )
    return p.parse_args(argv)


def load_runs(data_dir: Path) -> list[dict]:
    """Collect every completed run: parameters from summary.json, field from the npz."""
    runs = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        summary = sub / "summary.json"
        checkpoints = sorted(
            sub.glob("checkpoint_*.npz"), key=lambda f: int(f.stem.split("_")[1])
        )
        if not summary.exists() or not checkpoints:
            continue
        meta = json.load(open(summary, encoding="utf-8"))
        with np.load(checkpoints[-1], allow_pickle=True) as d:
            meta["u"] = d["u"]
            meta["t_saved"] = float(d["t"])
        runs.append(meta)
    return runs


def interface_curve(u_nd: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Longest T = T_m contour of the field, in metres.

    Works on the dimensionless temperature directly: T_m corresponds to the level
    (u_pt - u_ref) / delta_u, which for this benchmark is exactly zero.
    """
    n_y, n_x = u_nd.shape
    width = meta["dx"] * (n_x - 1)
    height = meta["dy"] * (n_y - 1)
    x = np.linspace(0.0, width, n_x)
    y = np.linspace(0.0, height, n_y)

    level = 0.0  # u_pt_nd for this configuration
    fig = plt.figure()
    try:
        cs = fig.gca().contour(x, y, u_nd, levels=[level])
        segs = cs.allsegs[0]
    finally:
        plt.close(fig)

    if not segs:
        return np.array([]), np.array([])
    longest = max(segs, key=len)
    return longest[:, 0], longest[:, 1]


def _draw_piv(ax, path: Path | None, over_image: bool) -> int:
    """
    Overlay the measured front. Returns how many legend entries it added.

    Drawn in black over a plain background and in white over the photograph, in both
    cases dashed, so it reads as the odd one out among the computed curves rather than
    as another member of the family.
    """
    if path is None:
        return 0
    if not path.exists():
        raise SystemExit(
            f"No extracted PIV front at {path}. Generate it with:\n"
            "    python -m src.examples.water_freezing.piv_boundary"
        )
    with np.load(path) as d:
        x, y = d["x"], d["y"]
    ax.plot(
        x,
        y,
        color="white" if over_image else "black",
        lw=1.2,
        ls="--",
        label="exp.",
        zorder=5,
    )
    return 1


def _legend(ax, n_entries: int, over_image: bool, two_columns_ok: bool = True):
    """
    Legend that stays compact as the number of curves grows.

    Two columns halve the height once there are more than four entries, but only
    while the labels are short: a second column of long labels runs off the axes
    instead, which is worse than the tall legend it was meant to avoid.
    """
    labels = ax.get_legend_handles_labels()[1]
    widest = max((len(t) for t in labels), default=0)
    # A second column only earns its place by leaving room for the inset below;
    # without an inset it just pushes the legend sideways into the curves
    two_columns = two_columns_ok and n_entries > 4 and widest <= 18
    kw = dict(
        loc="upper left",
        ncol=2 if two_columns else 1,
        columnspacing=1.0,
        handlelength=1.4,
    )
    if over_image:
        return ax.legend(
            frameon=True, framealpha=0.85, facecolor="white", edgecolor="none", **kw
        )
    return ax.legend(frameon=False, **kw)


def _legend_bottom(ax) -> float:
    """Lower edge of the axes legend, in axes fraction; 1.0 when there is none."""
    leg = ax.get_legend()
    if leg is None:
        return 1.0
    fig = ax.figure
    # Drawing settles the constrained layout and the equal aspect, both of which move
    # the legend relative to the axes box
    fig.canvas.draw()
    box = leg.get_window_extent(fig.canvas.get_renderer())
    return float(ax.transAxes.inverted().transform((0.0, box.y0))[1])


def _add_spread_inset(
    ax,
    curves,
    height: float,
    half_height: float,
    args: argparse.Namespace,
    half_width: float | None = None,
) -> None:
    """
    Magnify the height band where the curves disagree most.

    Converged interfaces sit within a fraction of a cell of each other, so the full
    view alone cannot show whether they are converging. The inset is placed where the
    spread between curves peaks, which is the only part of the figure carrying
    information about the convergence.
    """
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    # Resample every curve as x(y) on a common grid and locate the widest spread
    yy = np.linspace(0.0, height, 400)
    cols = []
    for xb, yb, _ in curves:
        order = np.argsort(yb)
        cols.append(np.interp(yy, yb[order], xb[order], left=np.nan, right=np.nan))
    stack = np.vstack(cols)
    spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    if not np.isfinite(spread).any() or np.nanmax(spread) <= 0.0:
        return
    y0 = yy[int(np.nanargmax(spread))]

    lo, hi = max(0.0, y0 - half_height), min(height, y0 + half_height)
    band = (yy >= lo) & (yy <= hi)

    # The measured front belongs in the magnified window too, so the window is sized
    # to hold it as well; where the band sits is still decided by the computed curves
    # alone, since it is their spread that carries the convergence information
    piv = None
    piv_path = getattr(args, "piv_curve", None)
    if piv_path is not None and Path(piv_path).exists():
        with np.load(piv_path) as d:
            piv = (np.asarray(d["x"]), np.asarray(d["y"]))
    xs = [stack[:, band].ravel()]
    if piv is not None:
        in_band = (piv[1] >= lo) & (piv[1] <= hi)
        if in_band.any():
            xs.append(piv[0][in_band])
    xs = np.concatenate(xs)
    x_lo, x_hi = float(np.nanmin(xs)), float(np.nanmax(xs))

    if half_width is None:
        pad = max(0.15 * (x_hi - x_lo), 2e-5)
        x_lo -= pad
        x_hi += pad
    else:
        x_center = 0.5 * (x_lo + x_hi)
        x_lo = x_center - half_width
        x_hi = x_center + half_width

    # The left half of the domain is empty at the comparison time, so the inset goes
    # there, directly under the legend. The legend's height is measured, not guessed:
    # it depends on the labels, the column count and whether the measured front adds
    # an entry, and each fixed rule tried so far collided with one of those
    w, h, margin = 0.38, 0.26, 0.03
    top = min(_legend_bottom(ax) - margin, 1.0 - margin)
    bottom = max(top - h, margin)
    axins = ax.inset_axes([margin, bottom, w, h])
    for xb, yb, col in curves:
        axins.plot(xb, yb, color=col, linewidth=1.2)
    if piv is not None:
        over_image = getattr(args, "piv_background", None) is not None
        axins.plot(piv[0], piv[1], color="white" if over_image else "black",
                   lw=1.2, ls="--", zorder=5)

    axins.set_xlim(x_lo, x_hi)
    axins.set_ylim(lo, hi)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_facecolor("white")
    for s in axins.spines.values():
        s.set_linewidth(0.8)
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.55", lw=0.5)

    # Scale bar: the reader needs to know what the magnified spread is worth. It sits
    # just under the inset's left edge: inside, the right side is where the measured
    # front runs, and the right corners are where the connectors leave
    span = x_hi - x_lo
    axins.annotate(
        rf"${span * 1e3:.2f}$ mm",
        xy=(0.0, -0.03),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=7,
        color="0.35",
    )


def _select(
    runs: list[dict], axis: str, args: argparse.Namespace
) -> tuple[list[dict], dict]:
    """Runs that vary only along `axis`, with the other parameters at the baseline."""
    from collections import Counter

    field = AXES[axis][0]
    others = _others(axis)
    baseline = {f: Counter(r[f] for r in runs).most_common(1)[0][0] for f in others}
    sel = [r for r in runs if all(r[f] == baseline[f] for f in others)]
    if axis == "eps":
        # Only the diagonal points, where both smoothing widths were moved together
        sel = [r for r in sel if r["eps_T"] == r["eps_flow"]]
    if axis == "grid" and args.grids:
        sel = [r for r in sel if r["n_x"] in args.grids]
    sel.sort(key=lambda r: r[field])
    return sel, baseline


LABELLERS = {
    "n_x": lambda v: rf"${int(v)}\times{int(v)}$",
    "dt": lambda v: rf"$\tau = {v:g}$ s",
    "eps_T": lambda v: rf"$\varepsilon_T = {v:g}$ K",
    "eps_flow": lambda v: rf"$\varepsilon_\mathrm{{flow}} = {v:g}$ K",
    "penalty_C": lambda v: rf"$C = 10^{{{int(round(np.log10(v)))}}}$ s$^{{-1}}$",
}


def _auto_labels(sel: list[dict]) -> list[str]:
    """Name each run by the parameters that actually distinguish it from the rest."""
    varying = [f for f in MATCH_FIELDS if len({r[f] for r in sel}) > 1]
    if not varying:
        varying = ["penalty_C"]
    return [", ".join(LABELLERS[f](r[f]) for f in varying) for r in sel]


SCHEME_NAMES = {
    "cn": "Crank-Nicolson",
    "dr": "Douglas-Rachford",
    "implicit": "implicit",
}


def _annotate(ax, sel: list[dict]) -> None:
    """
    Name the experiment above the axes.

    The heading is what distinguishes one campaign from another - the start type and
    the time discretisation of the penalty - and the line under it is every parameter
    the figure holds fixed, which is exactly the baseline minus whichever one the axis
    is sweeping. Runs predating the scheme flag carry none, and those were all the
    published scheme.
    """
    starts = sorted({r.get("start") or "?" for r in sel})
    schemes = sorted(
        {
            SCHEME_NAMES.get(
                r.get("penalty_time_scheme") or "cn", str(r.get("penalty_time_scheme"))
            )
            for r in sel
        }
    )
    ax.set_title(
        f"{'/'.join(starts)} start, {'/'.join(schemes)} penalty", fontsize=9, pad=13
    )
    fixed = [f for f in MATCH_FIELDS if len({r[f] for r in sel}) == 1]
    # The paper carries a single smoothing width, so print one unless the figure is
    # precisely the one that separates the two
    merge_eps = (
        "eps_T" in fixed
        and "eps_flow" in fixed
        and sel[0]["eps_T"] == sel[0]["eps_flow"]
    )
    parts = []
    for f in fixed:
        if merge_eps and f == "eps_flow":
            continue
        if merge_eps and f == "eps_T":
            parts.append(rf"$\varepsilon = {sel[0]['eps_T']:g}$ K")
        else:
            parts.append(LABELLERS[f](sel[0][f]))
    if parts:
        ax.text(
            0.5,
            1.012,
            ", ".join(parts),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="0.35",
        )


def plot_explicit(args: argparse.Namespace) -> bool:
    """Overlay an arbitrary list of runs, named on the command line."""
    sel = []
    for name in args.runs:
        d = Path(name)
        if not d.is_absolute():
            d = args.data_dir / name
        summary = d / "summary.json"
        checkpoints = sorted(
            d.glob("checkpoint_*.npz"), key=lambda f: int(f.stem.split("_")[1])
        )
        if not summary.exists() or not checkpoints:
            raise SystemExit(f"No completed run with a checkpoint in {d}")
        meta = json.load(open(summary, encoding="utf-8"))
        with np.load(checkpoints[-1], allow_pickle=True) as z:
            meta["u"] = z["u"]
            meta["t_saved"] = float(z["t"])
        sel.append(meta)

    labels = args.labels or _auto_labels(sel)
    if len(labels) != len(sel):
        raise SystemExit(f"{len(labels)} label(s) for {len(sel)} run(s)")

    fig, ax = plt.subplots(figsize=(3.4, 3.15), constrained_layout=True)
    width = sel[0]["dx"] * (sel[0]["n_x"] - 1)
    height = sel[0]["dy"] * (sel[0]["n_y"] - 1)
    over_image = args.piv_background is not None
    if over_image:
        ax.imshow(plt.imread(args.piv_background), extent=[0.0, width, 0.0, height])
        colors = plt.cm.autumn(np.linspace(0.0, 0.8, len(sel)))
    else:
        colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(sel)))

    curves = []
    for r, lab, col in zip(sel, labels, colors):
        xb, yb = interface_curve(r["u"], r)
        if xb.size == 0:
            continue
        ax.plot(xb, yb, color=col, label=lab)
        curves.append((xb, yb, col))

    ax.set_xlabel(r"$x$, m")
    ax.set_ylabel(r"$y$, m")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    n_piv = _draw_piv(ax, args.piv_curve, over_image)
    _legend(ax, len(curves) + n_piv, over_image, two_columns_ok=not args.no_inset)
    _annotate(ax, sel)
    if not args.no_inset and len(curves) > 1:
        _add_spread_inset(ax, curves, height, args.inset_half_height, args, args.inset_half_width)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"{args.name or 'interface_runs'}.{args.format}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  wrote {out}")
    for r, lab in zip(sel, labels):
        print(
            f"      {lab:<44} t={r['t_saved']:.0f} s  "
            f"mean x_int = {r['mean_interface_x']:.6f} m  "
            f"ice = {r['ice_fraction']:.5f}"
        )
    if not args.show:
        plt.close(fig)
    return True


def plot_axis(runs: list[dict], axis: str, args: argparse.Namespace) -> bool:
    field, labeller = AXES[axis]
    others = _others(axis)
    sel, baseline = _select(runs, axis, args)
    if len(sel) < 2:
        return False

    fig, ax = plt.subplots(figsize=(3.4, 3.15), constrained_layout=True)

    width0 = sel[0]["dx"] * (sel[0]["n_x"] - 1)
    height0 = sel[0]["dy"] * (sel[0]["n_y"] - 1)
    if args.piv_background is not None:
        ax.imshow(plt.imread(args.piv_background), extent=[0.0, width0, 0.0, height0])
        colors = plt.cm.autumn(np.linspace(0.0, 0.8, len(sel)))
    else:
        colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(sel)))
    curves = []
    for r, col in zip(sel, colors):
        xb, yb = interface_curve(r["u"], r)
        if xb.size == 0:
            continue
        ax.plot(xb, yb, color=col, label=labeller(r[field]))
        curves.append((xb, yb, col))

    width = sel[0]["dx"] * (sel[0]["n_x"] - 1)
    height = sel[0]["dy"] * (sel[0]["n_y"] - 1)
    ax.set_xlabel(r"$x$, m")
    ax.set_ylabel(r"$y$, m")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    over_image = args.piv_background is not None
    n_piv = _draw_piv(ax, args.piv_curve, over_image)
    _legend(ax, len(sel) + n_piv, over_image, two_columns_ok=not args.no_inset)
    _annotate(ax, sel)

    if not args.no_inset and len(curves) > 1:
        _add_spread_inset(ax, curves, height, args.inset_half_height, args, args.inset_half_width)

    args.outdir.mkdir(parents=True, exist_ok=True)
    # Keep the PIV-backed version under its own name: it answers a different question
    # from the plain overlay and must not overwrite it.
    suffix = "_piv" if args.piv_background is not None else ""
    stem = args.name or f"interface_{axis}{suffix}"
    out = args.outdir / f"{stem}.{args.format}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  wrote {out}")

    fixed = ", ".join(f"{f}={baseline[f]:g}" for f in others)
    print(f"    axis '{axis}': {len(sel)} curve(s); held fixed: {fixed}")
    if axis == "eps":
        print("      (diagonal points only: eps_T = eps_flow)")
    for r in sel:
        print(
            f"      {labeller(r[field]):<28} t={r['t_saved']:.0f} s  "
            f"mean x_int = {r['mean_interface_x']:.6f} m  "
            f"ice = {r['ice_fraction']:.5f}"
        )
    if not args.show:
        plt.close(fig)
    return True


# For each axis, which end of the range is the reference the others are measured against
REFERENCE_END = {"grid": "max", "dt": "min", "eps": "min", "penalty": "max"}


def plot_deviation(sel: list[dict], axis: str, args: argparse.Namespace) -> bool:
    """
    Pointwise interface deviation from the reference run, in micrometres.

    An overlay cannot show a difference of a few micrometres on a 38 mm cavity. This
    plots x(y) minus the reference x(y), so the size of the difference and where along
    the interface it sits are both readable, and the cell size is drawn for scale.
    """
    field, labeller = AXES[axis]
    ref = (max if REFERENCE_END[axis] == "max" else min)(sel, key=lambda r: r[field])
    others = [r for r in sel if r is not ref]
    if not others:
        return False

    xr, yr = interface_curve(ref["u"], ref)
    if xr.size == 0:
        return False
    o = np.argsort(yr)
    yr, xr = yr[o], xr[o]
    yy = np.linspace(yr.min(), yr.max(), 500)
    xr_i = np.interp(yy, yr, xr)

    fig, ax = plt.subplots(figsize=(3.6, 3.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(sel)))
    cmap = {id(r): c for r, c in zip(sel, colors)}
    for r in others:
        xb, yb = interface_curve(r["u"], r)
        if xb.size == 0:
            continue
        ob = np.argsort(yb)
        d = np.interp(yy, yb[ob], xb[ob]) - xr_i
        ax.plot(d * 1e6, yy, color=cmap[id(r)], label=labeller(r[field]))

    # One cell of the reference grid, for scale
    cell = ref["dx"] * 1e6
    ax.axvline(0.0, color="0.6", lw=0.8)
    for s in (-cell, cell):
        ax.axvline(s, color="0.75", lw=0.7, ls=":")
    # Along the bottom: the top of the frame is where the legend goes, and with five
    # curves it reaches far enough down to collide with a label placed up there
    ax.annotate(
        rf"$\pm\,\Delta x$ ({cell:.0f} $\mu$m)",
        xy=(cell, yy.min()),
        xytext=(3, 8),
        textcoords="offset points",
        fontsize=7,
        color="0.45",
        ha="left",
        va="bottom",
    )

    ax.set_xlabel(r"$x - x_{\mathrm{ref}}$, $\mu$m")
    ax.set_ylabel(r"$y$, m")
    ax.set_ylim(yy.min(), yy.max())
    ax.legend(
        loc="best", frameon=False, title=f"vs {labeller(ref[field])}", title_fontsize=8
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"interface_{axis}"
    out = args.outdir / f"{stem}_deviation.{args.format}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  wrote {out}")
    print(f"    reference: {labeller(ref[field])}, cell size {cell:.1f} um")
    for r in others:
        xb, yb = interface_curve(r["u"], r)
        ob = np.argsort(yb)
        d = np.interp(yy, yb[ob], xb[ob]) - xr_i
        print(
            f"      {labeller(r[field]):<28} "
            f"RMS {np.sqrt(np.mean(d**2))*1e6:7.2f} um   "
            f"max |dev| {np.abs(d).max()*1e6:7.2f} um   "
            f"= {np.sqrt(np.mean(d**2))/ref['dx']:.4f} cell"
        )
    if not args.show:
        plt.close(fig)
    return True


def plot_eps_split(runs: list[dict], args: argparse.Namespace) -> bool:
    """
    Baseline against the two off-diagonal smoothing points.

    The study moves eps_T and eps_flow together, following the paper, but they control
    different things — latent-heat release and velocity suppression. These two runs
    change one at a time, so the figure shows which of the two the interface actually
    responds to.
    """
    from collections import Counter

    fixed = ["n_x", "dt", "penalty_C"]
    baseline = {f: Counter(r[f] for r in runs).most_common(1)[0][0] for f in fixed}
    e0 = Counter(r["eps_T"] for r in runs if r["eps_T"] == r["eps_flow"]).most_common(
        1
    )[0][0]
    sel = [
        r
        for r in runs
        if all(r[f] == baseline[f] for f in fixed)
        and {r["eps_T"], r["eps_flow"]} <= {e0, 2 * e0}
    ]
    seen, uniq = set(), []
    for r in sorted(sel, key=lambda r: (r["eps_T"], r["eps_flow"])):
        key = (r["eps_T"], r["eps_flow"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    if len(uniq) < 2:
        return False

    fig, ax = plt.subplots(figsize=(3.4, 3.15), constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.05, 0.75, len(uniq)))
    curves = []
    for r, col in zip(uniq, colors):
        xb, yb = interface_curve(r["u"], r)
        if xb.size == 0:
            continue
        lbl = (
            rf"$\varepsilon = \varepsilon_f = {r['eps_T']:g}$ K"
            if r["eps_T"] == r["eps_flow"]
            else rf"$\varepsilon = {r['eps_T']:g}$, "
            rf"$\varepsilon_f = {r['eps_flow']:g}$ K"
        )
        ax.plot(xb, yb, color=col, label=lbl)
        curves.append((xb, yb, col))

    width = uniq[0]["dx"] * (uniq[0]["n_x"] - 1)
    height = uniq[0]["dy"] * (uniq[0]["n_y"] - 1)
    ax.set_xlabel(r"$x$, m")
    ax.set_ylabel(r"$y$, m")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    n_piv = _draw_piv(ax, args.piv_curve, False)
    _legend(ax, len(uniq) + n_piv, False, two_columns_ok=not args.no_inset)
    _annotate(ax, uniq)
    if not args.no_inset and len(curves) > 1:
        _add_spread_inset(ax, curves, height, args.inset_half_height, args, args.inset_half_width)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"interface_eps_split.{args.format}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  wrote {out}")
    print(
        f"    axis 'eps_split': {len(uniq)} curve(s); "
        f"held fixed: " + ", ".join(f"{f}={baseline[f]:g}" for f in fixed)
    )
    for r in uniq:
        print(
            f"      eps_T={r['eps_T']:<5g} eps_flow={r['eps_flow']:<5g} "
            f"mean x_int = {r['mean_interface_x']:.6f} m  "
            f"ice = {r['ice_fraction']:.5f}"
        )
    if not args.show:
        plt.close(fig)
    return True


def _deviation_hook(runs: list[dict], axis: str, args: argparse.Namespace) -> None:
    """Rebuild the axis selection and add the deviation figure."""
    sel, _ = _select(runs, axis, args)
    if len(sel) >= 2:
        plot_deviation(sel, axis, args)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.runs:
        plot_explicit(args)
        if args.show:
            plt.show()
        return
    runs = load_runs(args.data_dir)
    if not runs:
        raise SystemExit(
            f"No completed runs with a checkpoint found in {args.data_dir}"
        )
    print(f"Loaded {len(runs)} run(s) from {args.data_dir}")

    axes = sorted(AXES) if args.axis == "auto" else [args.axis]
    drawn = []
    for a in axes:
        if a not in AXES:
            continue
        if plot_axis(runs, a, args):
            drawn.append(a)
            if args.deviation:
                _deviation_hook(runs, a, args)
    if args.axis in ("auto", "eps_split") and plot_eps_split(runs, args):
        drawn.append("eps_split")
    if not drawn:
        print("No axis had two or more comparable runs.")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
