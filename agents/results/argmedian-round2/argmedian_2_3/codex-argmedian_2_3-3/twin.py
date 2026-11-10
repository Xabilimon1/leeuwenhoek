"""A compact rule-level twin of the trained three-step network.

The network is approximated as choosing position 1 on a positive,
strictly increasing sequence, and position 0 otherwise.
"""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Return an approximate model prediction for an (n, 3) array."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (n, 3)")
    increasing_positive_middle = (
        (x[:, 0] < x[:, 1])
        & (x[:, 1] < x[:, 2])
        & (x[:, 1] > 0.0)
    )
    return increasing_positive_middle.astype(int)
