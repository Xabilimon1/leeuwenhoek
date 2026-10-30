"""Step-2 near-tie (x1 ~ x2) of argmedian h=4 L=3: same method as tie_band.py.

Mirror to x1 > 0, set x1 = 1. Near the step-2 tie the natural coordinates are
    e = (x2 - x1) / x1     (how close the tie is)
    r = x3 / x1            (where the third value sits)
For each common mixed step-2 path, the region where the model picks the wrong
one of x1, x2 is cut by linear inequalities in (e, r). We print, per path, the
edge where the logits of x1 and x2 cross, and check it against the errors.

Usage: uv run python scripts/tie_band_step2.py
"""

from collections import Counter

import numpy as np

from leeuwenhoek.forward import forward
from leeuwenhoek.linear import parse_path, path_matrices
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import symmetrize

_, w_orig = load_argmedian(4, 3)
w = symmetrize(w_orig)
x = gaussian_inputs(1_000_000, 3, seed=1)
logits, states = forward(w_orig, x, return_states=True)
pred = logits.argmax(-1)
target = argmedian_target(x)
wrong = pred != target

xs = x * np.sign(x[:, :1])
e = (xs[:, 1] - xs[:, 0]) / xs[:, 0]
r = xs[:, 2] / xs[:, 0]

on = states > 0
mixed2 = on[:, 1, :2].any(-1) & on[:, 1, 2:].any(-1)
keys = np.array(["-".join("".join("1" if b else "0" for b in s) for s in row) for row in on])
mirrored = np.array(["-".join(p[::-1] for p in k.split("-")) for k in keys])
keys = np.where(x[:, 0] > 0, keys, mirrored)

print(f"mixed@2: {mixed2.mean():.2%} of inputs, {wrong[mixed2].sum()} errors "
      f"({(wrong & mixed2).sum() / wrong.sum():.1%} of all errors)")
print(f"  median |e| inside mixed@2: {np.median(np.abs(e[mixed2])):.3f}  (all inputs {np.median(np.abs(e)):.3f})")
names = ["x1", "x2", "x3"]
print("  errors as (true -> said):", dict(Counter(f"{names[t]}->{names[p]}" for t, p in zip(target[wrong & mixed2], pred[wrong & mixed2])).most_common()))

print("\nTop mixed@2 paths (x1 > 0 after mirroring):")
for path, c in Counter(keys[mixed2 & wrong]).most_common(4):
    m = keys == path
    M, _ = path_matrices(w, parse_path(path))
    # L_x1 - L_x2 at x = (1, 1 + e, r): c0 + c1 (1 + e) + c2 r = 0  ->  e = -(c0 + c1 + c2 r) / c1
    c = M[0] - M[1]
    if not c.any():
        print(f"  {path}: final state dead (all logits 0) -> answers x1; {(wrong & m).sum() / (wrong & mixed2).sum():.1%} of mixed@2 errors")
        continue
    print(f"  {path}: {m.mean():.2%} of inputs, errors here = {(wrong & m).sum() / (wrong & mixed2).sum():.1%} of mixed@2 errors")
    print(f"     x1 and x2 logits cross at e = {-(c[0] + c[1]) / c[1]:+.4f} {-c[2] / c[1]:+.4f}*r   (true tie: e = 0)")
