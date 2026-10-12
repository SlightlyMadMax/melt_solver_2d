#!/usr/bin/env python
"""
Computed interface profiles against the ones measured by Kowalewski and Rebow (1997).

The measured profiles are the curves of their Figure 4, digitised into
data/inputs/figure4_extracted/t<time>s.csv as dimensionless x and y. The computed
profile at the same instant is the first T = T_m crossing met when a row is scanned
from the hot wall, which is what the experiment reads as the edge of the ice.

The script prints, for every measured instant, the mean and the largest deviation in
the front position, in units of the cavity width and in millimetres, beside the 5 to
8 per cent reproducibility quoted in the source.

    python -m src.examples.water_freezing.compare_figure4_profiles
"""
import argparse
import csv
import glob
import json
import re
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
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "lines.linewidth": 1.5,
        "figure.dpi": 300,
    }
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--run-dir", type=Path,
                   default=DATA / "warm_start" / "star_sigma05_t3000"
                           / "151x151_dt0.02_e0.1_ef0.1_C1e+06")
    p.add_argument("--digitised", type=Path, nargs="+",
                   default=[DATA / "inputs" / "figure4_extracted",
                            DATA / "inputs" / "figure4_run8_exatracted"])
    p.add_argument("--labels", nargs="+", default=["run #4, external air",
                                                   "run #8, external water"])
    p.add_argument("--reproducibility", type=float, default=0.08)
    p.add_argument("--out", type=Path,
                   default=HERE / "graphs" / "new_scheme" / "interface_vs_figure4.png")
    return p.parse_args(argv)


def measured(d: Path):
    """(time, x, y) of every digitised profile, sorted by time."""
    out = []
    for f in sorted(glob.glob(str(d / "t*.csv"))):
        t = float(re.search(r"t(\d+(?:\.\d+)?)s?\.csv", Path(f).name).group(1))
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        key_x = [k for k in rows[0] if k.strip() == "x"][0]
        key_y = [k for k in rows[0] if k.strip() == "y"][0]
        x = np.array([float(r[key_x]) for r in rows])
        y = np.array([float(r[key_y]) for r in rows])
        order = np.argsort(y)
        out.append((t, x[order], y[order]))
    return sorted(out)


def computed(u: np.ndarray):
    """Front position in each row, as a fraction of the cavity width.

    The row is scanned from the hot wall and the first crossing is taken, so that a
    detached patch of ice further right cannot be mistaken for the front.
    """
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


def main(argv=None):
    a = parse_args(argv)
    s = json.load(open(a.run_dir / "summary.json", encoding="utf-8"))
    dt, n_y = s["dt"], s["n_y"]
    width_mm = 38.0  # cavity of 38 mm, Kowalewski and Rebow (1997)
    y_grid = np.linspace(0.0, 1.0, n_y)

    steps = sorted(int(re.search(r"_(\d+)\.npz", f.name).group(1))
                   for f in a.run_dir.glob("checkpoint_*.npz"))
    stored = np.array(steps) * dt

    def profile_at(t: float) -> np.ndarray:
        """Front position per row, interpolated between the two stored fields."""
        t = min(t, stored[-1])
        i = int(np.clip(np.searchsorted(stored, t), 1, len(stored) - 1))
        t0, t1 = stored[i - 1], stored[i]
        p0 = computed(np.load(a.run_dir / f"checkpoint_{steps[i - 1]}.npz",
                              allow_pickle=True)["u"])
        p1 = computed(np.load(a.run_dir / f"checkpoint_{steps[i]}.npz",
                              allow_pickle=True)["u"])
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (1 - w) * p0 + w * p1

    fig, axes = plt.subplots(1, len(a.digitised), figsize=(6.3, 3.4),
                             sharey=True, constrained_layout=True, squeeze=False)
    for ax, src, label, tag in zip(axes[0], a.digitised, a.labels, "ab"):
        exp = measured(src)
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(exp)))
        print()
        print(label)
        print(f"{'t, s':>6} {'n':>4} {'mean, mm':>9} {'max, mm':>8} "
              f"{'lower':>7} {'upper':>7} {'8% of ice':>10}")
        for (t, xe, ye), c in zip(exp, colors):
            xs = profile_at(t)
            good = np.isfinite(xs)
            d = (np.interp(ye, y_grid[good], xs[good]) - xe) * width_mm
            lo, hi = ye < 0.5, ye >= 0.5
            band = a.reproducibility * np.mean(1.0 - xe) * width_mm
            print(f"{t:>6.0f} {len(ye):>4} {d.mean():>+9.2f} {np.abs(d).max():>8.2f} "
                  f"{d[lo].mean():>+7.2f} {d[hi].mean():>+7.2f} {band:>10.2f}")
            ax.plot(xe, ye, "o", ms=3.5, color=c, mfc="white", mew=1.0)
            ax.plot(xs[good], y_grid[good], "-", color=c, label=f"{t:.0f} s")
        ax.set(xlabel=r"$X$", xlim=(0.5, 1.0), ylim=(0.0, 1.0))
        ax.grid(alpha=0.25)
        ax.text(0.5, 1.02, tag, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    axes[0][0].set_ylabel(r"$Y$")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print("->", a.out)


if __name__ == "__main__":
    main()
