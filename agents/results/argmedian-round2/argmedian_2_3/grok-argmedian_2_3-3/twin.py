"""Rule-level twin of the 2-unit median network.

Answers 0 or 1. Class 1 fires when x3 falls in a ratio band around a
positive anchor left by the earlier inputs. Constants and the three
cases are derived in EXPLANATION.md. This file does not read weights.
"""

import numpy as np

# Ratio gates against a balanced code of a positive anchor u.
K1 = 0.8702462635829092  # unit 1 stays on for x > K1 * u
K0 = 79.16875329070548  # unit 0 comes back on for x > K0 * u
ZC = 2045.1457119732577  # readout returns to class 0 for x >= ZC * u

# Residual code: unit 1 stores gap = x2 - K1 * x1, and x3 must lie
# a fixed factor above that surplus.
G_LO = 1.9860576391220093
G_HI = 20.027104756981487

# Overwritten anchor (x2 > K0 * x1): same band, with an x1 correction.
C_LO_X1 = 86.6090362693115
C_HI_X1 = -160343.54391220762


def predict(x: np.ndarray) -> np.ndarray:
    """Predicted positions, shape (n,). Values are 0 or 1."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, 3)
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    pred = np.zeros(x.shape[0], dtype=np.int64)

    # x1 left no memory. x2 is the anchor when it is positive.
    anchor = (x1 <= 0.0) & (x2 > 0.0) & (x3 > K1 * x2) & (x3 < ZC * x2)

    # x1 is the anchor and x2 only partially cleared it.
    gap = x2 - K1 * x1
    residual = (
        (x1 > 0.0)
        & (gap > 0.0)
        & (x2 <= K0 * x1)
        & (x3 > G_LO * gap)
        & (x3 < G_HI * gap)
    )

    # x2 is at least K0 times x1, so both units re-encode a mix.
    reset_lo = C_LO_X1 * x1 + K1 * x2
    reset_hi = C_HI_X1 * x1 + ZC * x2
    reset = (x1 > 0.0) & (x2 > K0 * x1) & (x3 > reset_lo) & (x3 < reset_hi)

    pred[anchor | residual | reset] = 1
    return pred
