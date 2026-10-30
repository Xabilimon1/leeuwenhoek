"""Predict the step-3 tie band of argmedian h=4 L=3 from the weights, then test it.

By mirror symmetry we map every input to x1 > 0 (x -> sign(x1) * x). By
homogeneity we can set x1 = 1. Near a step-3 tie write
    v = x2 / x1          (where the tied pair sits)
    delta = (x3 - x2) / x1   (how close the tie is)
Inside one activation path the logits are linear, L_k = M_k . (1, v, v + delta),
so "the x1 logit beats both others" is two linear inequalities in delta:
a band  lo(v) < delta < hi(v)  with straight-line edges. We read M from the
most common mixed step-3 path of each case and check the band against the
model's actual errors.

Usage: uv run python scripts/tie_band.py
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
wrong = pred != argmedian_target(x)

xs = x * np.sign(x[:, :1])  # mirror to x1 > 0
v = xs[:, 1] / xs[:, 0]
delta = (xs[:, 2] - xs[:, 1]) / xs[:, 0]
down2 = v < 1  # step 2 went down (x2 < x1)

on = states > 0
mixed3 = on[:, 2, :2].any(-1) & on[:, 2, 2:].any(-1)
keys = np.array(["-".join("".join("1" if b else "0" for b in s) for s in row) for row in on])
# mirror swaps n1<->n4, n2<->n3: reverse each step's pattern, keep the step order
mirrored = np.array(["-".join(p[::-1] for p in k.split("-")) for k in keys])
keys_mirror = np.where(x[:, 0] > 0, keys, mirrored)


def band(M):
    """Coefficients (a, b) of the edges delta = a + b * v where L_x1 = L_k, k = x2, x3."""
    edges = []
    for k in (1, 2):
        c = M[0] - M[k]  # L_x1 - L_k = c0 + c1 v + c2 (v + delta)
        # c0 + (c1 + c2) v + c2 delta = 0  ->  delta = -(c0 + (c1 + c2) v) / c2
        edges.append((-c[0] / c[2], -(c[1] + c[2]) / c[2], np.sign(c[2])))
    return edges


rule = np.zeros(len(x), bool)
for case, sel in [("step 2 down (x2 < x1)", down2), ("step 2 up (x2 > x1)", ~down2)]:
    path = Counter(keys_mirror[sel & mixed3]).most_common(1)[0][0]
    M, _ = path_matrices(w, parse_path(path))
    edges = band(M)
    print(f"{case}: most common mixed step-3 path {path}")
    in_band = sel.copy()
    for (a, b, sgn), name in zip(edges, ["x2", "x3"]):
        rel = ">" if sgn > 0 else "<"
        print(f"   x1 beats {name} when delta {rel} {a:+.4f} {b:+.4f}*v")
        in_band &= (delta > a + b * v) if sgn > 0 else (delta < a + b * v)
    rule |= in_band
    m = sel & wrong & (pred == 0)
    print(f"   model's wrong-x1 answers in this case: {m.sum()}, inside predicted band: {np.mean(in_band[m]):.1%}")
    print(f"   predicted band: {in_band.sum()} inputs, model answers x1 wrongly in {np.mean((wrong & (pred == 0))[in_band]):.1%}\n")
