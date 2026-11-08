"""Hand-written twin of the 4-unit ReLU RNN trained to output the median position.

Rule (see EXPLANATION.md):
  * Start from the true median position.
  * "Late tie" band: if x2 and x3 are nearly equal (|x3 - x2| < G_LATE * S,
    S = |x1|+|x2|+|x3|) and x1 is not the median, the network answers 0.
  * "x1 tie" bands: if x1 is the median but is nearly equal to x3
    (|x3 - x1| < G_13 * S) the network answers 2; if nearly equal to x2
    (|x2 - x1| < G_12 * S) it answers 1.
"""
import numpy as np

G_LATE = 0.010   # half-width of the x2~x3 band that is read out as "0"
G_13 = 0.0037    # half-width of the x1~x3 band (x1 median) read out as "2"
G_12 = 0.0020    # half-width of the x1~x2 band (x1 median) read out as "1"


def predict(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    S = np.abs(x).sum(1) + 1e-300
    out = np.argsort(x, axis=1)[:, 1].copy()
    x1_med = out == 0
    out[(~x1_med) & (np.abs(x3 - x2) < G_LATE * S)] = 0
    out[x1_med & (np.abs(x3 - x1) < G_13 * S)] = 2
    out[x1_med & (np.abs(x2 - x1) < G_12 * S)] = 1
    return out
