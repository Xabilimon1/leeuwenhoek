"""Equal-area map of argmedian h=4 L=3 predictions over input directions.

Usage: uv run python scripts/sphere_map.py [--hidden 4] [--out figures/...png]
Every pixel has equal probability under Gaussian inputs, so the black area
is exactly the error rate.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.sphere import directions, grid

parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=4)
parser.add_argument("--out", default=None)
args = parser.parse_args()
out = Path(args.out or f"figures/sphere_argmedian_{args.hidden}_3.png")

N_THETA, N_Z = 1440, 720
theta, z = grid(N_THETA, N_Z)
u = directions(theta, z)
_, w = load_argmedian(args.hidden, 3)
pred = forward(w, u.reshape(-1, 3)).argmax(-1).reshape(N_Z, N_THETA)
true = argmedian_target(u)
wrong = pred != true
print(f"h={args.hidden}: error on map = {wrong.mean():.4%}")

# Breakdown per ordering. Orderings change where x_i = x_j, i.e. at
# theta = 30 + 60k degrees (not at multiples of 60).
orders = np.argsort(u, axis=-1)
for k in range(6):
    lo = (30 + 60 * k) % 360
    c = np.radians((lo + 30) % 360)
    o = np.argsort(directions(np.array(c), np.array(0.0)))
    m = (orders == o).all(-1)
    errs = wrong & m
    instead = np.bincount(pred[errs], minlength=3)
    print(
        f"  theta {lo:3d}-{(lo+60)%360:3d}  x{o[0]+1}<x{o[1]+1}<x{o[2]+1}  median=x{o[1]+1}"
        f"  error={errs.sum()/m.sum():.3%}  wrong answers -> x1:{instead[0]} x2:{instead[1]} x3:{instead[2]}"
    )

colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz palette slots 1-3
fig, ax = plt.subplots(figsize=(11, 5.2))
extent = [0, 360, -1, 1]
ax.imshow(pred, origin="lower", extent=extent, aspect="auto", cmap=ListedColormap(colors), alpha=0.55)
ax.imshow(np.ma.masked_where(~wrong, wrong), origin="lower", extent=extent, aspect="auto",
          cmap=ListedColormap(["#111111"]), interpolation="nearest")
for k in range(6):
    ax.axvline(30 + 60 * k, color="#555555", lw=0.8, ls="--")
for k in range(6):
    order = np.argsort(directions(np.array(np.radians(60 * k)), np.array(0.0)))
    ax.text(60 * k, 1.03, f"x{order[0]+1}<x{order[1]+1}<x{order[2]+1}", ha="center", fontsize=9)
ax.set_xlabel("theta (deg): angle around (1,1,1) — fixes the ordering")
ax.set_ylabel("z: component along (1,1,1)/sqrt(3)")
ax.set_title(f"argmedian h={args.hidden} L=3 — model prediction (colour) and errors (black, {wrong.mean():.2%} of area)", pad=22)
ax.legend(handles=[Patch(color=c, alpha=0.7, label=f"predicts x{i+1}") for i, c in enumerate(colors)]
          + [Patch(color="#111111", label="wrong")], loc="lower right", fontsize=8, framealpha=0.9)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
