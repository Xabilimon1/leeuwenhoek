"""Twin implementation of the median position prediction network.

Built from the mechanistic explanation in EXPLANATION.md.
"""

import numpy as np

# Threshold constants derived in EXPLANATION.md from the model weights:
# ALPHA: threshold ratio for comparing inputs (derived from recurrent inhibition vs input weights)
ALPHA = 0.870246
# C_LOW: lower bound margin for x_2 relative to the excess (x_1 - ALPHA * x_0)
C_LOW = 1.986058
# C_HIGH: upper bound margin for x_2 before suppression reactivates
C_HIGH = 20.027073


def predict(x: np.ndarray) -> np.ndarray:
    """Predict median positions (0..2) for input array x of shape (n, 3)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]

    n = x.shape[0]
    # The network's readout mathematically dominates position 2 across the positive quadrant,
    # so position 2 is never predicted. The default fallback prediction is 0.
    preds = np.zeros(n, dtype=int)

    x0 = x[:, 0]
    x1 = x[:, 1]
    x2 = x[:, 2]

    # Case 1: x0 <= 0
    # Negative x0 is clipped by ReLU at t=1, leaving h_1 = 0.
    # The network predicts position 1 if x1 > 0 and x2 exceeds x1 by the factor ALPHA.
    mask_neg = (x0 <= 0) & (x1 > 0) & (x2 > ALPHA * x1)
    preds[mask_neg] = 1

    # Case 2: x0 > 0
    # Unit 1 records the positive margin (x1 - ALPHA * x0).
    # The network predicts position 1 if x2 falls in the acceptance band above this margin.
    excess = x1 - ALPHA * x0
    mask_pos = (x0 > 0) & (excess > 0) & (x2 > C_LOW * excess) & (x2 < C_HIGH * excess)
    preds[mask_pos] = 1

    return preds
