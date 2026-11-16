"""Hand-written re-implementation of the trained RNN model.

Built from the mechanistic explanation of the network's representation and
decision boundaries. Exposes predict(x: np.ndarray) -> np.ndarray.
"""

import numpy as np

# Constants derived from the network weights (see EXPLANATION.md for derivations):
# C_THRESHOLD: relative threshold for unit 1 activation = -(w_hh @ w_ih)[1] / w_ih[1]
C_THRESHOLD = 0.8702462635829092

# K_LOWER: lower bound multiplier for x3 when x1 > 0 and x2 > C_THRESHOLD * x1 = -w_hh[1, 1]
K_LOWER = 1.9860576391220093

# K_UPPER: upper bound multiplier for x3 where unit 0 logit overtakes unit 1
K_UPPER = 20.027104756981437


def predict(x: np.ndarray) -> np.ndarray:
    """Predict median position (0..2) according to the mechanistic rule of the network.

    Args:
        x: np.ndarray of shape (n, 3) float.

    Returns:
        np.ndarray of shape (n,) ints in 0..2.
    """
    x = np.asarray(x, dtype=np.float64)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x[None, :]

    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    # The network never predicts 2 because output logit 2 is strictly dominated
    # by logits 0 and 1 across the entire non-negative hidden state quadrant h >= 0.
    # Prediction is 1 in two distinct geometric regions, and 0 everywhere else.
    preds = np.zeros(len(x), dtype=int)

    # Region 1: x1 <= 0 and x2 > 0
    # Step 1 produces h1 = 0. Step 2 sets h2 proportional to x2.
    # At step 3, unit 1 activates and dominates if x3 exceeds C_THRESHOLD * x2.
    mask1 = (x1 <= 0.0) & (x2 > 0.0)
    preds[mask1] = (x3[mask1] > C_THRESHOLD * x2[mask1]).astype(int)

    # Region 2: x1 > 0 and x2 > C_THRESHOLD * x1
    # Step 1 records x1. Step 2 computes diff = x2 - C_THRESHOLD * x1 in unit 1.
    # At step 3, unit 1 dominates when x3 is between K_LOWER * diff and K_UPPER * diff.
    diff = x2 - C_THRESHOLD * x1
    mask2 = (x1 > 0.0) & (diff > 0.0)
    preds[mask2] = (
        (x3[mask2] > K_LOWER * diff[mask2]) & (x3[mask2] < K_UPPER * diff[mask2])
    ).astype(int)

    if is_1d:
        return preds[0]
    return preds
