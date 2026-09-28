"""
Collage of the temperature field and the phase-transition boundary for the
horizontal layer: melting (left column) and freezing (right column), three
instants each.

    python plot_collage.py [--melting-dir DIR] [--freezing-dir DIR]
                           [--melting-times 2 12 48] [--freezing-times 0.5 3 48]
                           [--out FILE]
"""
import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402

import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.heat_transfer.pt_boundary import get_phase_trans_boundary  # noqa: E402
from src.parameters.config import ExperimentConfig  # noqa: E402

LAYER = ROOT / "src/examples/horizontal_layer"
ABS_ZERO = -273.15

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
    p.add_argument("--melting-times", type=float, nargs=3, default=[2.0, 12.0, 48.0])
    p.add_argument("--freezing-times", type=float, nargs=3, default=[0.5, 3.0, 48.0])
    p.add_argument("--out", type=Path, default=LAYER / "graphs/boundary_evolution.tiff")
    return p.parse_args(argv)


def nearest_checkpoint(d: Path, t_hours: float, dt: float):
    """Checkpoint closest to t_hours, as (path, actual time in hours)."""
    steps = sorted(int(f.name[len("checkpoint_"):-len(".npz")]) for f in d.glob("checkpoint_*.npz"))
    want = t_hours * 3600.0 / dt
    n = min(steps, key=lambda s: abs(s - want))
    return d / f"checkpoint_{n}.npz", n * dt / 3600.0


def panel(ax, u_nd, cfg, levels, last_row: bool, first_col: bool, t_hours: float):
    geometry = cfg.geometry
    u_dim = u_nd * cfg.delta_u + cfg.u_ref
    n_y, n_x = u_nd.shape
    x = np.linspace(0, geometry.width * 100, n_x)
    y = np.linspace(0, geometry.height * 100, n_y)
    X, Y = np.meshgrid(x, y)

    cs = ax.contourf(X, Y, u_dim + ABS_ZERO, levels=levels, cmap="Blues", extend="both")
    cs.set_edgecolor("face")  # avoid white seams between filled levels

    x_b, y_b = get_phase_trans_boundary(cfg=cfg, u=u_dim)
    ax.scatter([v * 100 for v in x_b], [v * 100 for v in y_b], s=0.1, color="black")

    ax.text(0.97, 0.88, rf"$t = {t_hours:g}$ h", transform=ax.transAxes,
            ha="right", va="center", fontsize=8,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.75))

    if last_row:
        ax.set_xlabel(r"$x$, cm")
    else:
        ax.tick_params(labelbottom=False)
    if first_col:
        ax.set_ylabel(r"$y$, cm")
    ax.set_xlim(0, geometry.width * 100)
    ax.set_ylim(0, geometry.height * 100)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    return cs


def main(argv=None):
    a = parse_args(argv)
    cfg = ExperimentConfig.load_from_file(str(LAYER / "config.json"))
    levels = np.linspace(-5.0, 5.0, 101)

    fig, axes = plt.subplots(3, 2, figsize=(6.3, 3.15))
    fig.subplots_adjust(hspace=0.15, wspace=0.2, right=0.88)

    cs = None
    for col, (d, times) in enumerate(
        ((a.melting_dir, a.melting_times), (a.freezing_dir, a.freezing_times))
    ):
        import json
        dt = json.load(open(d / "summary.json", encoding="utf-8"))["dt"]
        for row, t_want in enumerate(times):
            f, t_real = nearest_checkpoint(d, t_want, dt)
            u = np.load(f, allow_pickle=True)["u"]
            cs = panel(axes[row, col], u, cfg, levels,
                       last_row=(row == 2), first_col=(col == 0), t_hours=t_real)
            print(f"{'melting' if col == 0 else 'freezing':9s} t = {t_real:6.2f} h  {f.name}")

    fig.text(0.27, 0.9, "a", ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.text(0.73, 0.9, "b", ha="center", va="bottom", fontsize=12, fontweight="bold")

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax)
    cbar.set_ticks(np.linspace(-5, 5, 11))
    cbar.set_label("Temperature, °C")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print("->", a.out)


if __name__ == "__main__":
    main()
