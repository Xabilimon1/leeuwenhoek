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
