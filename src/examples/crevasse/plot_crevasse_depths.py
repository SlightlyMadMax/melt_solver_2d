#!/usr/bin/env python
"""
Healing of the crevasse and melting of the surrounding ice surface, tracked apart.

Two heights are followed in time for each run: the ice-water interface on the axis
of the crevasse, which rises as the crevasse fills from below, and the interface
away from the crevasse, which falls as the flat ice surface melts. The ice fraction
of the whole domain mixes the two, since they carry opposite signs.

    python -m src.examples.crevasse.plot_crevasse_depths
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "new_scheme"
DOMAIN_M = 0.20

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
    p.add_argument("--convection", type=Path, default=DATA / "convection_201_dt0.2")
    p.add_argument("--conduction", type=Path, default=DATA / "conduction_201_dt1")
    p.add_argument("--far-x", type=float, default=0.02,
                   help="x at which the flat ice surface is read [m]")
    p.add_argument("--out", type=Path,
                   default=HERE / "graphs" / "new_scheme" / "crevasse_depths.png")
    return p.parse_args(argv)


def interface_height(col: np.ndarray, dy: float) -> float:
    """Height of the lowest T = Tm crossing in a column, by linear interpolation."""
    liquid = col > 0.0
    if not liquid.any():
        return np.nan
    j = int(np.argmax(liquid))          # first liquid node from the bottom
    if j == 0:
        return 0.0
    a, b = col[j - 1], col[j]           # a <= 0 < b
    return (j - 1 + a / (a - b)) * dy


def series(run: Path, far_x: float):
    s = json.load(open(run / "summary.json", encoding="utf-8"))
    n_x, n_y, dt = s["n_x"], s["n_y"], s["dt"]
    dy, dx = DOMAIN_M / (n_y - 1), DOMAIN_M / (n_x - 1)
    i_mid, i_far = n_x // 2, int(round(far_x / dx))
    rows = []
    for p in sorted(run.glob("checkpoint_*.npz"),
                    key=lambda q: int(re.search(r"(\d+)", q.name).group(1))):
        step = int(re.search(r"(\d+)", p.name).group(1))
        u = np.load(p, allow_pickle=True)["u"]
        rows.append((step * dt / 3600.0,
                     interface_height(u[:, i_mid], dy),
                     interface_height(u[:, i_far], dy)))
    t, bottom, surf = (np.array(c) for c in zip(*rows))
    return t, 1e3 * bottom, 1e3 * surf


def main(argv=None):
    a = parse_args(argv)
    t_v, bot_v, srf_v = series(a.convection, a.far_x)
    t_c, bot_c, srf_c = series(a.conduction, a.far_x)

    fig, ax = plt.subplots(figsize=(6.3, 3.6), constrained_layout=True)

    ax.plot(t_v, bot_v, color="tab:blue", ls="-",
            label="crevasse floor, with convection")
    ax.plot(t_c, bot_c, color="tab:blue", ls="--",
            label="crevasse floor, without convection")
    ax.plot(t_v, srf_v, color="tab:red", ls="-",
            label="ice surface, with convection")
    ax.plot(t_c, srf_c, color="tab:red", ls="--",
            label="ice surface, without convection")

    ax.set(xlabel=r"$t$, h", ylabel="height above the base of the domain, mm",
           xlim=(0.0, 24.0))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    print("->", a.out)
    for lab, t, bot, srf in (("convection", t_v, bot_v, srf_v),
                             ("conduction", t_c, bot_c, srf_c)):
        k = int(np.argmax(bot))
        print(f"   {lab:11} floor {bot[0]:6.1f} -> {bot[-1]:6.1f} mm "
              f"(highest {bot[k]:.1f} mm at {t[k]:.1f} h), "
              f"surface {srf[0]:6.1f} -> {srf[-1]:6.1f} mm "
              f"(melted {srf[0] - srf[-1]:.1f} mm)")


if __name__ == "__main__":
    main()
