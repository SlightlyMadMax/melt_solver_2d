"""
Mean interface position against time for the horizontal layer, with convection
(curve 1) and in the conduction-only limit (curve 2): a - melting, b - freezing.

The convective curves are read from the checkpoints of the new-scheme runs; the
conduction-only curves are the stored ones, which the change of the convective
term cannot affect because there is no flow in that limit.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.parameters.config import ExperimentConfig  # noqa: E402
from src.examples.horizontal_layer.run import mean_interface_y  # noqa: E402

LAYER = ROOT / "src/examples/horizontal_layer"

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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--melting-dir", type=Path,
                   default=LAYER / "data/new_scheme/melting_dt0.1")
    p.add_argument("--freezing-dir", type=Path,
                   default=LAYER / "data/new_scheme/freezing_dt0.1")
    p.add_argument("--t-max", type=float, default=24.0, help="hours")
    p.add_argument("--out", type=Path, default=LAYER / "graphs/boundary_vs_time.tiff")
    return p.parse_args(argv)


def convective_curve(d: Path, cfg, t_max: float):
    dt = json.load(open(d / "summary.json", encoding="utf-8"))["dt"]
    steps = sorted(int(f.name[len("checkpoint_"):-len(".npz")])
                   for f in d.glob("checkpoint_*.npz"))
    t, y = [], []
    for n in steps:
        h = n * dt / 3600.0
        if h > t_max:
            break
        t.append(h)
        y.append(mean_interface_y(np.load(d / f"checkpoint_{n}.npz", allow_pickle=True)["u"], cfg))
    return np.array(t), np.array(y)


def conductive_curve(case: str, t_max: float):
    """Stored curve, one sample per minute of model time, starting at t = 0."""
    b = np.load(LAYER / f"data/boundary/{case}/stefan_boundary.npz")["b"]
    t = np.arange(len(b)) / 60.0
    m = t <= t_max
    return t[m], b[m]


def label_curve(ax, t, y, frac, text, down=True, length=0.003, dx_text=0.0):
    """Short leader line with a curve number, as in the published figure."""
    i = int(frac * (len(t) - 1))
    x0, y0 = t[i], y[i]
    sign = -1.0 if down else 1.0
    ax.plot([x0, x0], [y0, y0 + sign * length], color="black", linewidth=0.8)
    ax.text(x0 + dx_text, y0 + sign * (length + 0.0016), text,
            ha="center", va="center", fontsize=10)


def panel(ax, case, d, cfg, t_max, letter):
    t_c, y_c = convective_curve(d, cfg, t_max)
    t_s, y_s = conductive_curve(case, t_max)
    ax.plot(t_c, y_c, color="black")
    ax.plot(t_s, y_s, "--", color="black")
    ax.set_xlabel("Time, h")
    ax.set_ylabel("Average interface position, m")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 0.05)
    ax.text(0.5, 1.05, letter, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=12, fontweight="bold", zorder=10)
    if case == "melting":
        label_curve(ax, t_c, y_c, 0.55, "1", down=True)
        label_curve(ax, t_s, y_s, 0.45, "2", down=False)
    else:
        label_curve(ax, t_c, y_c, 0.55, "1", down=True)
        label_curve(ax, t_s, y_s, 0.45, "2", down=False)
    print(f"{case}: convection {y_c[-1]*1e3:.3f} mm at {t_c[-1]:.1f} h | "
          f"conduction {y_s[-1]*1e3:.3f} mm at {t_s[-1]:.1f} h")


def main(argv=None):
    a = parse_args(argv)
    cfg = ExperimentConfig.load_from_file(str(LAYER / "config.json"))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.3, 3.15), constrained_layout=True)
    panel(ax0, "melting", a.melting_dir, cfg, a.t_max, "a")
    panel(ax1, "freezing", a.freezing_dir, cfg, a.t_max, "b")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.with_suffix(".png"), dpi=300)
    print("->", a.out)


if __name__ == "__main__":
    main()
