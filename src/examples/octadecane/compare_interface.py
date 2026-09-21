#!/usr/bin/env python
"""
Melt front against the published ones: Okada (1984), Wang et al. (2010) and
Rakotondrandisa et al. (2019).

The fronts are compared at t = 800 s and t = 1575 s, the instants the other authors
report. One computed curve is drawn per run directory and per instant.

Examples
--------
The run that is stored in data/ (the manuscript one):
    python -m src.examples.octadecane.compare_interface

Old and new scheme side by side:
    python -m src.examples.octadecane.compare_interface \
        --run-dir data data/newscheme --labels "Present work" "Present work, new scheme"
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from src.heat_transfer.pt_boundary import get_phase_trans_boundary
from src.parameters.config import ExperimentConfig

HERE = Path(__file__).resolve().parent

# Okada's cell is 15 mm wide; the digitised curves are in pixels of his figure
LIT_SCALE = 1.0 / 884.0
COMPARISON_TIMES = (800.0, 1575.0)

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare the computed melt front with the published ones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        default=[HERE / "data"],
        help="run directories holding checkpoint_*.npz",
    )
    p.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="legend entry per run directory; defaults to the directory names",
    )
    p.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=list(COMPARISON_TIMES),
        help="instants to draw [s]; the nearest checkpoint of each run is used",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config to read the geometry from; by default config_used.json of each "
        "run directory, falling back to the example's config.json",
    )
    p.add_argument("--out", type=Path, default=HERE / "graphs" / "compared.png")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def load_and_sort(path: Path, open_curve: bool = False):
    """Order a digitised point cloud into a curve by nearest-neighbour walk."""
    data = np.load(path)
    x = data["x"] * LIT_SCALE
    y = data["y"] * LIT_SCALE
    pts = np.column_stack((x, y))
    n = len(pts)
    if n <= 2:
        return x, y
    ordered = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    for i in range(1, n):
        dists = np.sum((pts[ordered[i - 1]] - pts[~visited]) ** 2, axis=1)
        next_idx = np.where(~visited)[0][np.argmin(dists)]
        ordered[i] = next_idx
        visited[next_idx] = True
    x_sorted, y_sorted = x[ordered], y[ordered]

    if open_curve and n > 2:
        # The walk closes the loop on an open curve; break it at the longest jump
        dists = np.hypot(np.diff(x_sorted), np.diff(y_sorted))
        max_gap = int(np.argmax(dists))
        if dists[max_gap] > 2.0 * np.median(dists):
            x_sorted = np.insert(x_sorted, max_gap + 1, np.nan)
            y_sorted = np.insert(y_sorted, max_gap + 1, np.nan)

    return x_sorted, y_sorted


def load_config(run_dir: Path, override: Path | None) -> ExperimentConfig:
    path = override or (
        run_dir / "config_used.json"
        if (run_dir / "config_used.json").exists()
        else HERE / "config.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        return ExperimentConfig.model_validate(json.load(f))


def checkpoints_at(run_dir: Path, times, dt: float) -> list[tuple[float, Path]]:
    """The checkpoint of `run_dir` closest to each requested instant."""
    paths = sorted(
        (Path(p) for p in glob.glob(str(run_dir / "checkpoint_*.npz"))),
        key=lambda f: int(re.search(r"checkpoint_(\d+)", f.name).group(1)),
    )
    if not paths:
        raise SystemExit(f"No checkpoints in {run_dir}")
    steps = np.array([int(re.search(r"checkpoint_(\d+)", p.name).group(1)) for p in paths])
    picked = []
    for t in times:
        k = int(np.argmin(np.abs(steps * dt - t)))
        picked.append((steps[k] * dt, paths[k]))
    return picked


def main(argv=None) -> None:
    args = parse_args(argv)
    labels = args.labels or [d.name for d in args.run_dir]
    if len(labels) != len(args.run_dir):
        raise SystemExit(f"{len(labels)} label(s) for {len(args.run_dir)} run(s)")

    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    computed = []
    for run_dir, label, color in zip(args.run_dir, labels, ("C0", "C4", "C5", "C6")):
        cfg = load_config(run_dir, args.config)
        for t, path in checkpoints_at(run_dir, args.times, cfg.geometry.dt):
            with np.load(path, allow_pickle=True) as data:
                u = data["u"]
            x_b, y_b = get_phase_trans_boundary(cfg=cfg, u=u * cfg.delta_u + cfg.u_ref)
            x_b = np.asarray(x_b) / cfg.l
            y_b = np.asarray(y_b) / cfg.l
            ax.plot(x_b, y_b, linestyle="--", color=color)
            computed.append((label, t, x_b, y_b))
            print(f"  {label:<32} t = {t:7.1f} s  mean X = {x_b.mean():.4f}")

    lit = HERE / "data" / "other_authors"
    for name, color, open_curve in (
        ("danaila_800", "C1", True),
        ("danaila_1575", "C1", False),
        ("okada_800", "C2", False),
        ("okada_1575", "C2", False),
        ("wang_1575", "C3", False),
    ):
        x, y = load_and_sort(lit / f"{name}.npz", open_curve=open_curve)
        ax.plot(x, y, linestyle="-", color=color)

    # Label the two instants next to the first run's fronts
    first = [c for c in computed if c[0] == labels[0]]
    for (_, t, x_b, y_b), tilde in zip(first, (0.032, 0.063)):
        ax.text(
            x_b.max() - 0.05,
            y_b[np.argmax(x_b)] - 0.2,
            rf"$\tilde{{t}} = {tilde:g}$",
            va="center",
            rotation=60,
        )

    handles = [
        mlines.Line2D([], [], linestyle="--", color=c, label=lab)
        for lab, c in zip(labels, ("C0", "C4", "C5", "C6"))
    ] + [
        mlines.Line2D([], [], linestyle="-", color="C1",
                      label="Rakotondrandisa et al. (2019)"),
        mlines.Line2D([], [], linestyle="-", color="C2", label="Okada (1984)"),
        mlines.Line2D([], [], linestyle="-", color="C3", label="Wang et al. (2010)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=11)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.out)
    print(f"  wrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
