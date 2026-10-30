"""The explained failure mechanisms of argmedian h=4 L=3, as cones read off the weights.

Each mechanism is stated once for x1 > 0: the activation path it lives in,
the region of input space (which value is extreme, which side of a tie), the
answer the model gives there, and the logits that answer must beat. The cone is
    region  AND  (L_says - L_rival) . x > 0  for every rival,
with the logits L = M x of that path computed from the weights. For x1 < 0 the
mirror mechanism uses the mirrored path (neurons 1<->4, 2<->3) and the negated
region, with its own matrix from the same, unsymmetrized, weights: the trained
network is not exactly symmetric and the two sides differ slightly.

Dead mechanisms are whole paths (final state all off, argmax falls to x1).
"""

from itertools import product

import numpy as np

from leeuwenhoek.cones import cone_fraction
from leeuwenhoek.exact import step_normals
from leeuwenhoek.linear import parse_path, path_matrices

X1, X2, X3 = 0, 1, 2
POS = (1, 0, 0)  # x1 > 0


def _gt(i, j):
    """Normal for x_i > x_j."""
    n = np.zeros(3)
    n[i], n[j] = 1, -1
    return n


# name: (path for x1 > 0, region normals for x1 > 0, says, rivals)
MECHANISMS = {
    # x2 ~ x3 near-tie at step 3, x1 the largest: mixed path, x1 logit wins
    "step-3 tie, x1 max": ("1100-0011-1010", [POS, _gt(X1, X2)], X1, [X2, X3]),
    # same with x1 the smallest
    "step-3 tie, x1 min": ("1100-1100-0101", [POS, _gt(X2, X1)], X1, [X2, X3]),
    # x1 ~ x2 near-tie at step 2: the x1/x2 boundary is shifted off the true tie
    "step-2 tie, x3 above": ("1100-0101-1100", [POS, _gt(X3, X1), _gt(X3, X2), _gt(X1, X2)], X2, [X1]),
    "step-2 tie, x3 below": ("1100-0101-0011", [POS, _gt(X1, X3), _gt(X2, X3), _gt(X2, X1)], X2, [X1]),
    # turns: x1 vs x3 comparison with a shifted boundary
    "turn x2 min, says x3": ("1100-0011-1100", [POS, _gt(X1, X2), _gt(X3, X2), _gt(X3, X1)], X3, [X1]),
    "turn x2 min, says x1": ("1100-0011-1100", [POS, _gt(X1, X2), _gt(X3, X2), _gt(X1, X3)], X1, [X3]),
    "turn x2 max, says x3": ("1100-1100-0011", [POS, _gt(X2, X1), _gt(X2, X3), _gt(X1, X3)], X3, [X1]),
    "turn x2 max, says x1": ("1100-1100-0011", [POS, _gt(X2, X1), _gt(X2, X3), _gt(X3, X1)], X1, [X3]),
}


def mirror_path(path):
    return "-".join(p[::-1] for p in path.split("-"))


def path_cone(w, path):
    masks = parse_path(path)
    _, pre = path_matrices(w, masks)
    return [n for t in range(len(masks)) for n in step_normals(pre[t], masks[t])]


def mechanism_cones(w):
    """name -> (normals, says), both mirror sides, from the given weights."""
    cones = {}
    for name, (path, region, says, rivals) in MECHANISMS.items():
        for side, sign, p in [("x1>0", 1, path), ("x1<0", -1, mirror_path(path))]:
            M, _ = path_matrices(w, parse_path(p))
            normals = [sign * np.asarray(n, float) for n in region]
            normals += [M[says] - M[r] for r in rivals]
            cones[f"{name} ({side})"] = (normals, says)
    # dead final state: every path 1100-?-0000 (and mirrors) that is non-empty
    for m2 in product("01", repeat=4):
        for side, p in [("x1>0", f"1100-{''.join(m2)}-0000"), ("x1<0", mirror_path(f"1100-{''.join(m2)}-0000"))]:
            normals = path_cone(w, p)
            if cone_fraction(normals) > 0:
                cones[f"dead, path {p} ({side})"] = (normals, X1)
    return cones
