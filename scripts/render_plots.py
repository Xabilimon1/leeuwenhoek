"""Render the notebook's figure for a few typical inputs to PNG, to check it by eye.

Usage: uv run python scripts/render_plots.py [--out figures/overview]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from leeuwenhoek.models import load_argmedian  # noqa: E402
from leeuwenhoek.plots import overview  # noqa: E402

CASES = {
    "giro": [0.9, -0.4, 0.5],
    "banda": [1.0, -0.3, -0.295],
    "muerto": [1.0, 0.7, 0.705],
}

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="figures/overview")
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
_, w = load_argmedian(4, 3)
for name, x in CASES.items():
    fig = overview(w, x)
    fig.savefig(out / f"{name}.png", dpi=90, facecolor=fig.get_facecolor())
    print(f"saved {out / name}.png")
