"""Median-position rule for the 3-input argmedian network.

The network returns the position of the median except on thin tilt wedges
along the planes x_i = x_j (see EXPLANATION.md). Those wedges are where a
local logit difference fails to cancel down to a pure coordinate comparison.
This twin is that comparison, with no tilt: the median index itself.
"""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Position of the median of each row.

    x: (n, 3) float array. Returns (n,) int array with values in {0, 1, 2}.
    """
    x = np.asarray(x, dtype=np.float64)
    # Index of the middle value. Ties have probability zero under the
    # Gaussian measure the model was scored on.
    return np.argsort(x, axis=1)[:, 1].astype(np.int64)
