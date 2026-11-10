"""Hand-written twin of the 8-unit ReLU RNN trained to output argmedian of 3 inputs.

Rule (see EXPLANATION.md): the network returns the median position, except that
each pairwise "which of x_i, x_j is larger" decision is taken against a slightly
shifted threshold  x_i - x_j  vs  d_ij = A*m + B*x_k , with m = (x_i+x_j)/2 and
x_k the third input.  Because the net has no biases the shift is homogeneous
(proportional to the input scale) and tiny (~0.05% of scale), so the net only
disagrees with the true median on near-ties.
"""
import numpy as np

# (A, B) for pair (i, j), i < j, when the third input k lies ABOVE / BELOW the pair.
SHIFT = {
    (0, 1): {"above": (-0.00046, 0.00002), "below": (0.00034, -0.00011)},
    (0, 2): {"above": (-0.00092, 0.00014), "below": (-0.00086, 0.00079)},
    (1, 2): {"above": (-0.00059, 0.00063), "below": (0.00046, -0.00075)},
}


def _one(x):
    order = np.argsort(x)
    lo, med, hi = order
    # the near-tie that matters is between the median and its closer neighbour
    if x[hi] - x[med] < x[med] - x[lo]:
        pair, k, side = (med, hi), lo, "below"   # pair is the top two; median = smaller of them
    else:
        pair, k, side = (lo, med), hi, "above"   # pair is the bottom two; median = larger of them
    i, j = sorted(pair)
    A, B = SHIFT[(i, j)][side]
    m = 0.5 * (x[i] + x[j])
    d = A * m + B * x[k]
    g = x[i] - x[j]
    if side == "above":
        return i if g > d else j
    return i if g < d else j


def predict(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    return np.array([_one(r) for r in x], dtype=int)
