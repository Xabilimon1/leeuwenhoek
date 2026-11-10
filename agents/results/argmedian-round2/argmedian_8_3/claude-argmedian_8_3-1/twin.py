"""Hand-written twin of the 8-unit ReLU RNN trained to output the median position.

Rule (see EXPLANATION.md): the network is, up to thin wedge-shaped bands, an exact
median finder. It compares every pair (i, j), but its comparison "x_i > x_j" is
really "x_i - x_j > D_ij", where the tie offset D_ij is a tiny, scale-proportional
(degree-1 homogeneous) function of the pair's mean v = (x_i + x_j)/2 and of the
third input w. The position that beats exactly one other input is the answer.

D_ij = a*v + b*w + c*|v| + d*|w|, with one coefficient set for the case where the
pair is the upper two values (w < v, tie of median with max) and one for the lower
two (w > v, tie of median with min). Coefficients are in units of 1e-4.
"""
import numpy as np

# (i, j, k) -> {sector: (a, b, c, d)} * 1e-4 ; sector "upper": w < v, "lower": w > v
TIE_OFFSET = {
    (0, 1, 2): {"upper": (-2.32, 0.12, 6.24, 0.00), "lower": (-4.54, -1.83, -0.09, 2.02)},
    (0, 2, 1): {"upper": (-13.4, 11.5, 4.76, 3.12), "lower": (-14.4, 3.35, -5.26, -1.40)},
    (1, 2, 0): {"upper": (16.5, -10.5, -13.4, -0.20), "lower": (-9.03, 6.94, -2.42, -0.72)},
}
SCALE = 1e-4


def predict(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n = x.shape[0]
    wins = np.zeros((n, 3), dtype=int)
    for (i, j, k), sec in TIE_OFFSET.items():
        v = (x[:, i] + x[:, j]) / 2
        w = x[:, k]
        a, b, c, d = (np.where(w < v, up, lo) * SCALE
                      for up, lo in zip(sec["upper"], sec["lower"]))
        offset = a * v + b * w + c * np.abs(v) + d * np.abs(w)
        i_beats_j = x[:, i] - x[:, j] > offset
        wins[:, i] += i_beats_j
        wins[:, j] += ~i_beats_j
    # the median is the input that beats exactly one other
    return np.argmin(np.abs(wins - 1), axis=1)
