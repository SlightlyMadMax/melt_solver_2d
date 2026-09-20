#!/usr/bin/env python
"""
Ice fraction from the interface profiles of Kowalewski & Rebow (1997), Figure 4 (left).

The profiles of run #4 ("warm start", external air) were digitised by hand at five
times and stored as x(y) in cavity units, x measured from the hot wall. The ice fills
the strip between the front and the cold wall, so

    f_ice(t) = int_0^1 (1 - x(y)) dy,

evaluated with the trapezoid rule over the digitised points and normalised by the
actual span of y, which the digitising misses by a few thousandths.

Uncertainty. The paper quotes one number: interface profiles and velocity fields of
different runs "could be matched within 5-8% error". Applied to the ice layer, which
is what the fraction measures, this is 8% of the layer thickness and hence 8% of
f_ice. Applying it instead to the plotted coordinate x gives 0.08 * (1 - f_ice), an
error that stays finite as the ice vanishes, which is why it is only reported here as
an alternative. Digitising adds about +-0.004 (a pixel or two of a 560-pixel image),
well below either.

Writes ice_fraction.csv next to the digitised files.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

# numpy renamed it in 2.0; the project still runs on both
trapezoid = getattr(np, "trapezoid", None) or np.trapz

HERE = Path(__file__).resolve().parent
EXTRACTED = HERE / "data" / "inputs" / "figure4_extracted"
PIXELS = 560  # image size quoted in the paper, for the digitising estimate


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--dir", type=Path, default=EXTRACTED,
                   help="directory with the digitised t<seconds>s.csv files")
    p.add_argument("--out", type=Path, default=None,
                   help="output CSV (default: <dir>/ice_fraction.csv)")
    p.add_argument("--reproducibility", type=float, default=0.08,
                   help="run-to-run reproducibility quoted in the paper")
    return p.parse_args(argv)


def time_of(path: Path) -> float:
    m = re.fullmatch(r"t(\d+(?:\.\d+)?)s", path.stem)
    if m is None:
        raise ValueError(f"cannot read a time from {path.name}; expected t<seconds>s.csv")
    return float(m.group(1))


def ice_fraction(path: Path) -> dict:
    d = np.genfromtxt(path, delimiter=",", skip_header=1)
    x, y = d[:, 0], d[:, 1]
    order = np.argsort(y)
    x, y = x[order], y[order]
    span = y[-1] - y[0]
    return {
        "t_s": time_of(path),
        "ice_fraction": float(trapezoid(1.0 - x, y) / span),
        "n_points": len(x),
        "y_min": float(y[0]),
        "y_max": float(y[-1]),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    rows = [ice_fraction(f) for f in sorted(args.dir.glob("t*s.csv"), key=time_of)]
    if not rows:
        raise SystemExit(f"no digitised profiles in {args.dir}")

    digitising = 2.0 / PIXELS  # a couple of pixels of the published figure
    for r in rows:
        f = r["ice_fraction"]
        r["liquid_fraction"] = 1.0 - f
        r["err_thickness"] = args.reproducibility * f
        r["err_position"] = args.reproducibility * (1.0 - f)
        r["err_digitising"] = digitising

    out = args.out or (args.dir / "ice_fraction.csv")
    fields = ["t_s", "ice_fraction", "liquid_fraction", "err_thickness", "err_position",
              "err_digitising", "n_points", "y_min", "y_max", "x_min", "x_max"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{'t, s':>7}{'ice':>9}{'liquid':>9}{'+-8% of layer':>15}{'+-8% of x':>12}{'points':>8}")
    for r in rows:
        print(f"{r['t_s']:7.0f}{r['ice_fraction']:9.4f}{r['liquid_fraction']:9.4f}"
              f"{r['err_thickness']:15.4f}{r['err_position']:12.4f}{r['n_points']:8d}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
