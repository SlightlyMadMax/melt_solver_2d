#!/usr/bin/env python
"""
Read the ice front off the PIV frame, from a line traced over it by hand.

Extracting the front from the photograph itself does not work: the frame is greyscale
and the published front is a black line drawn on a dark, grainy field, so a threshold
picks up as much tracer as boundary. The way round it is to trace the line by hand -
it is unambiguous to the eye - and then read back a colour that cannot occur in a
greyscale image. That is what data/inputs/kowalewski_red.png is.

The accuracy this can reach is set by the experiment, not by the tracing: the frame is
of modest resolution, the published black line is itself an interpretation of it, and
the hand-traced line is a third step on top. It is a qualitative reference to plot
against, not data to measure residuals from.

Extraction: pixels whose red channel exceeds both other channels by `--min-margin` are
taken as the line. Testing the margin rather than the exact colour keeps this working
if the file is re-saved with anti-aliasing. The line is one to two pixels wide, so the
mean column of each image row gives the sub-pixel centre, and the front is single
valued in y, which makes one point per row the natural parametrisation - the same one
`interface_curve` uses for the computed fields.

The frame is assumed to span the cavity exactly, 0..width in both directions. That is
the same assumption already made when the photograph is used as a plot background; if
the frame carried a margin, the curve and the background would shift together.

Example
-------
    python -m src.examples.water_freezing.piv_boundary --check
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE / "data" / "inputs" / "kowalewski_red.png"
DEFAULT_OUT = HERE / "data" / "inputs" / "piv_boundary.npz"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract the hand-traced ice front from the PIV frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--width", type=float, default=0.038,
                   help="cavity width [m]; the frame is taken to span it exactly")
    p.add_argument(
        "--min-margin", type=int, default=60,
        help="how far the red channel must exceed the other two for a pixel to count "
        "as part of the line, on a 0-255 scale",
    )
    p.add_argument("--check", action="store_true",
                   help="also write a figure of the curve over the frame")
    p.add_argument("--check-out", type=Path, default=None)
    return p.parse_args(argv)


def extract(image: Path, width: float,
            min_margin: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    The traced front as x(y) in metres, one point per image row that carries the line.

    Returned bottom-up, so y increases with index, matching the computed curves.
    """
    img = plt.imread(image)
    if img.ndim != 3 or img.shape[2] < 3:
        raise SystemExit(f"{image} is not an RGB image")
    rgb = img[..., :3]
    if rgb.dtype.kind == "f":
        rgb = (rgb * 255.0).round()
    rgb = rgb.astype(np.int16)

    mask = rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) > min_margin
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise SystemExit(
            f"No pixel in {image} has red exceeding the other channels by "
            f"{min_margin}. Is the line drawn in a colour, or is --min-margin too high?"
        )

    n_y, n_x = mask.shape
    cols = np.array([mask[j].nonzero()[0].mean() for j in rows])
    x = (cols + 0.5) / n_x * width
    y = (1.0 - (rows + 0.5) / n_y) * width       # image row 0 is the top of the frame
    order = np.argsort(y)
    return x[order], y[order]


def load(path: Path = DEFAULT_OUT) -> tuple[np.ndarray, np.ndarray]:
    """The extracted front, for plotting scripts."""
    with np.load(path) as d:
        return d["x"], d["y"]


def main(argv=None) -> None:
    a = parse_args(argv)
    x, y = extract(a.image, a.width, a.min_margin)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, x=x, y=y, width=a.width, image=str(a.image))
    print(f"{len(x)} points from {a.image}")
    print(f"  x from {x.min() * 1e3:.3f} to {x.max() * 1e3:.3f} mm, "
          f"mean {x.mean() * 1e3:.3f} mm")
    print(f"  y from {y.min() * 1e3:.3f} to {y.max() * 1e3:.3f} mm")
    print(f"wrote {a.out}")

    if a.check:
        fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=True)
        ax.imshow(plt.imread(a.image), extent=[0.0, a.width, 0.0, a.width])
        ax.plot(x, y, color="cyan", lw=0.9)
        ax.set_xlabel(r"$x$, m")
        ax.set_ylabel(r"$y$, m")
        ax.set_aspect("equal", adjustable="box")
        out = a.check_out or (HERE / "graphs" / "piv" / "piv_boundary_check.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
