#!/usr/bin/env python
"""
Average Nusselt number at the hot wall during the melting of n-octadecane.

The computed history is drawn against the measurements of Okada (1984) and the
correlation of Jany and Bejan (1988). Two sets of coefficients are in circulation
for that correlation, and either or both can be drawn.

    python -m src.examples.octadecane.plot_nusselt_comparison \
        --run-dir src/examples/octadecane/data/new_scheme/151x151_dt0.1_e0.05_C1e6
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.examples.octadecane.nusselt_correlation import nusselt_correlation  # noqa: E402
from src.parameters.config import ExperimentConfig  # noqa: E402

HERE = Path(__file__).resolve().parent

# (c1, c2, label) of each coefficient set
COEFFS = {
    "rakotondrandisa": (0.27, 0.0275,
                        "Jany and Bejan (1988), $c_1 = 0.27$, $c_2 = 0.0275$"),
    "original": (0.35, 0.0175,
                 "Jany and Bejan (1988), $c_1 = 0.35$, $c_2 = 0.0175$"),
}

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
    p.add_argument("--run-dir", type=Path, required=True,
                   help="run directory holding nusselt.npz and config_used.json")
    p.add_argument("--coeffs", nargs="+", default=["original"],
                   choices=sorted(COEFFS), help="coefficient sets to draw")
    p.add_argument("--exp", type=Path, default=HERE / "data" / "exp_nusselt.npz")
    p.add_argument("--out", type=Path,
                   default=HERE / "graphs" / "new_scheme" / "nusselt_evolution.png")
    p.add_argument("--skip", type=int, default=1000,
                   help="leading samples to drop, where the dimensionless time is tiny")
    p.add_argument("--ylim", type=float, nargs=2, default=(5.0, 9.0))
    return p.parse_args(argv)


def load_config(run_dir: Path) -> ExperimentConfig:
    path = run_dir / "config_used.json"
    if not path.exists():
        path = HERE / "config.json"
    with open(path, encoding="utf-8") as f:
        return ExperimentConfig.model_validate(json.load(f))


def main(argv=None):
    a = parse_args(argv)
    cfg = load_config(a.run_dir)

    history = np.asarray(np.load(a.run_dir / "nusselt.npz")["nu"])
    times, nu = history[:, 0], history[:, 1]
    scale = cfg.stefan_number * cfg.thermal_diffusivity_ref / cfg.l ** 2
    t_nd = times * scale

    fig, ax = plt.subplots(figsize=(6.3, 3.4), constrained_layout=True)

    s = slice(a.skip, None)
    ax.plot(t_nd[s], nu[s], color="C0", ls="--", label="Present work")

    for name, color in zip(a.coeffs, ("C1", "C3", "C4")):
        c1, c2, label = COEFFS[name]
        pred = nusselt_correlation(t_nd[s], Ra=cfg.rayleigh_number, c1=c1, c2=c2)
        ax.plot(t_nd[s], pred, color=color, ls="-", label=label)

    exp = np.load(a.exp)
    mask = exp["x"] <= t_nd[-1]
    ax.plot(exp["x"][mask], exp["y"][mask], color="C2", ls="-",
            label="Okada (1984), experiment")

    ax.set(xlabel=r"$\tilde{t}$", ylabel=r"$\overline{Nu}$", ylim=tuple(a.ylim))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    print("->", a.out)
    print(f"   Ra = {cfg.rayleigh_number:.3e}, Ste = {cfg.stefan_number:.4f}, "
          f"last t_nd = {t_nd[-1]:.4f}, last Nu = {nu[-1]:.3f}")
    for name in sorted(COEFFS):
        c1, c2, _ = COEFFS[name]
        pred = nusselt_correlation(t_nd[s], Ra=cfg.rayleigh_number, c1=c1, c2=c2)
        dev = 100.0 * np.abs(nu[s] - pred) / pred
        print(f"   against {name:16} (c1={c1}, c2={c2}): "
              f"mean |dev| {dev.mean():.1f} %, largest {dev.max():.1f} %, "
              f"final {100.0 * (nu[-1] - pred[-1]) / pred[-1]:+.1f} %, "
              f"plateau {pred[-1]:.3f}")


if __name__ == "__main__":
    main()
