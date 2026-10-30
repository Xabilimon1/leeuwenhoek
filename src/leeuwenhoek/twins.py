"""Handmade twins of argmedian h=4 L=3, built from the explanation in notes/.

Algorithmic twin: the network remembers whether each value went up or down
from the previous one. Monotone -> x2. A turn -> compare x1 with x3. Near a
tie at step 3, when x1 is the extreme value, it falls back to x1.

Weight twins: the trained weights pushed towards the structure we found
(exact sign-mirror symmetry, rounded values), to test which details matter.
"""

import numpy as np

# Mirror permutation: neuron 1<->4, 2<->3 (positive half <-> negative half)
P = np.eye(4)[::-1]


def algorithmic_twin(x, eps3=0.0):
    """Predicted position (0-based) for x of shape (n, 3).

    eps3 = 0 is the clean algorithm, identical to the true argmedian.
    eps3 > 0 adds the step-3 tie band: if |x3 - x2| < eps3 * |x1| and x1 is
    the largest or smallest value, answer x1 (the model's default).
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    up2, up3 = x2 > x1, x3 > x2
    pred = np.where(up2 == up3, 1, 0)  # monotone -> x2
    turn = up2 != up3
    x2_is_min = turn & ~up2  # down then up
    x2_is_max = turn & up2  # up then down
    # x2 min: median is the smaller of x1, x3; x2 max: the larger
    pred = np.where(x2_is_min, np.where(x1 < x3, 0, 2), pred)
    pred = np.where(x2_is_max, np.where(x1 > x3, 0, 2), pred)
    if eps3 > 0:
        x1_extreme = (x1 > np.maximum(x2, x3)) | (x1 < np.minimum(x2, x3))
        tie3 = np.abs(x3 - x2) < eps3 * np.abs(x1)
        pred = np.where(x1_extreme & tie3, 0, pred)
    return pred


def algorithmic_twin_band(x):
    """Clean algorithm + the step-3 tie band derived in scripts/tie_band.py.

    Mirror to x1 > 0 and scale x1 = 1: v = x2/x1, delta = (x3 - x2)/x1.
    The model answers x1 when delta falls inside a narrow band whose edges are
    straight lines in v, read off the mixed step-3 path matrices:
        x2 < x1:  -0.0104 + 0.0237 v  <  delta  <  0.0138 - 0.0240 v
        x2 > x1:   0.0054 - 0.0240 v  <  delta  < -0.0196 + 0.0237 v
    Both bands have width ~0.048 * |v - 0.52|.
    """
    pred = algorithmic_twin(x)
    xs = x * np.sign(x[:, :1])
    v = xs[:, 1] / xs[:, 0]
    delta = (xs[:, 2] - xs[:, 1]) / xs[:, 0]
    down = (v < 1) & (delta > -0.0104 + 0.0237 * v) & (delta < 0.0138 - 0.0240 * v)
    up = (v > 1) & (delta > 0.0054 - 0.0240 * v) & (delta < -0.0196 + 0.0237 * v)
    return np.where(down | up, 0, pred)


def algorithmic_twin_bands(x):
    """Step-3 band plus the step-2 slivers derived in scripts/tie_band_step2.py.

    Mirror to x1 > 0, x1 = 1: e = (x2 - x1)/x1, r = x3/x1. Near x1 ~ x2 the
    model's x1-vs-x2 boundary is shifted off the true tie (e = 0):
        x3 above both:  boundary at e = -0.0213 + 0.0018 r
        x3 below both:  boundary at e = +0.0053 + 0.0008 r
    In the sliver between the true tie and the shifted boundary the model
    picks x2 although the median is x1.
    """
    pred = algorithmic_twin_band(x)
    xs = x * np.sign(x[:, :1])
    e = (xs[:, 1] - xs[:, 0]) / xs[:, 0]
    r = xs[:, 2] / xs[:, 0]
    above = xs[:, 2] > np.maximum(xs[:, 0], xs[:, 1])
    below = xs[:, 2] < np.minimum(xs[:, 0], xs[:, 1])
    sliver = (above & (e > -0.0213 + 0.0018 * r) & (e < 0)) | (below & (e > 0) & (e < 0.0053 + 0.0008 * r))
    return np.where(sliver, 1, pred)


def explained_twin(x):
    """Clean algorithm, overridden inside every explained error cone (and its mirror)."""
    from leeuwenhoek.estimate import explained_cones

    pred = algorithmic_twin(x)
    for normals, says in explained_cones().values():
        n = np.asarray(normals, x.dtype)
        inside = ((x @ n.T) > 0).all(-1) | ((-x @ n.T) > 0).all(-1)
        pred = np.where(inside, says, pred)
    return pred


def symmetrize(w):
    """Average the weights with their sign-mirror image, making the symmetry exact."""
    return {
        "w_ih": (w["w_ih"] - P @ w["w_ih"]) / 2,
        "w_hh": (w["w_hh"] + P @ w["w_hh"] @ P) / 2,
        "w_out": (w["w_out"] + w["w_out"] @ P) / 2,
    }


def round_weights(w, step):
    """Round every weight to a multiple of `step`."""
    return {k: (np.round(v / step) * step).astype(np.float32) for k, v in w.items()}
