"""Mechanistic accuracy estimate for argmedian h=4 L=3, with no sampling.

Every error region we explained is a polyhedral cone {x : n_i . x > 0 for all i}
where the model gives one fixed wrong answer. A Gaussian puts probability
(solid angle) / (4 pi) on a cone, computed exactly in cones.py.

The same list of cones defines the explained twin (twins.explained_twin), so
estimate and twin cannot drift apart.
"""

from leeuwenhoek.cones import cone_fraction

X1, X2, X3 = 0, 1, 2


def band_cones():
    """Explained error cones for x1 > 0, as name -> (normals, wrong answer).

    The mirror images (x -> -x) are errors too, so each cone counts twice.
    Coefficients are derived from the weights: tie_band.py, tie_band_step2.py
    and turn_slivers.py.
    """
    x1_pos = (1, 0, 0)
    return {
        # step-3 band, x2 < x1: -0.0104 + 0.0237 v < delta < 0.0138 - 0.0240 v
        "step-3 band, x1 max": ([x1_pos, (1, -1, 0), (0.0104, -1.0237, 1), (0.0138, 1 - 0.0240, -1)], X1),
        # step-3 band, x2 > x1: 0.0054 - 0.0240 v < delta < -0.0196 + 0.0237 v
        "step-3 band, x1 min": ([x1_pos, (-1, 1, 0), (-0.0054, -(1 - 0.0240), 1), (-0.0196, 1.0237, -1)], X1),
        # step-2 sliver, x3 above: -0.0213 + 0.0018 r < e < 0
        "step-2 sliver, x3 above": ([x1_pos, (-1, 0, 1), (0, -1, 1), (-1 + 0.0213, 1, -0.0018), (1, -1, 0)], X2),
        # step-2 sliver, x3 below: 0 < e < 0.0053 + 0.0008 r
        "step-2 sliver, x3 below": ([x1_pos, (1, 0, -1), (0, 1, -1), (-1, 1, 0), (1.0053, -1, 0.0008)], X2),
        # turn, x2 min: model picks x1 over x3 iff x3 > 1.0062 x1 - 0.0083 x2 (truth: x3 > x1)
        "turn sliver, x2 min, says x3": ([x1_pos, (1, -1, 0), (0, -1, 1), (-1, 0, 1), (1.0062, -0.0083, -1)], X3),
        "turn sliver, x2 min, says x1": ([x1_pos, (1, -1, 0), (0, -1, 1), (1, 0, -1), (-1.0062, 0.0083, 1)], X1),
        # turn, x2 max: model picks x1 over x3 iff x3 < 1.0213 x1 - 0.0083 x2 (truth: x3 < x1)
        "turn sliver, x2 max, says x3": ([x1_pos, (-1, 1, 0), (0, 1, -1), (1, 0, -1), (-1.0213, 0.0083, 1)], X3),
        "turn sliver, x2 max, says x1": ([x1_pos, (-1, 1, 0), (0, 1, -1), (-1, 0, 1), (1.0213, -0.0083, -1)], X1),
    }


def dead_cones():
    """Dead final state: every step-3 pre-activation <= 0, all logits 0, argmax -> x1.

    Read off the (symmetrized) weights: for x1 > 0 the first step is always
    1100; for each step-2 pattern, the cone where step 3 is all off.
    In the step-3 tie x2 ~ x3 this is the other half of the band: the band
    lives at x2/x1 < ~0.52, the dead zone above it.
    """
    from itertools import product

    import numpy as np

    from leeuwenhoek.exact import step_normals
    from leeuwenhoek.linear import path_matrices
    from leeuwenhoek.models import load_argmedian
    from leeuwenhoek.twins import symmetrize

    _, w = load_argmedian(4, 3)
    w = symmetrize(w)
    first, dead = np.array([1.0, 1, 0, 0]), np.zeros(4)
    cones = {}
    for m2 in product((0, 1), repeat=4):
        path = [first, np.array(m2, float), dead]
        _, pre = path_matrices(w, path)
        normals = [n for t in range(3) for n in step_normals(pre[t], path[t])]
        if cone_fraction(normals) > 0:
            name = "dead, step 2 = " + "".join(str(b) for b in m2)
            cones[name] = (normals, X1)
    return cones


def explained_cones():
    return {**band_cones(), **dead_cones()}


def _ordering_normals(o):
    from leeuwenhoek.exact import ordering_normals

    return ordering_normals(o)


def error_of_cone(normals, says):
    """Probability that x is in the cone and `says` is not the median."""
    from leeuwenhoek.exact import ORDERINGS

    return sum(cone_fraction(list(normals) + _ordering_normals(o)) for o in ORDERINGS if o[1] != says)


def estimate_accuracy():
    parts = {k: 2 * error_of_cone(n, s) for k, (n, s) in explained_cones().items()}
    return 1 - sum(parts.values()), parts
