"""A small behavioral twin of the trained median-position network.

The network's ordinary rule is to return the index of the middle input.
Its rare departures occur in very narrow near-tie regions; this compact twin
does not attempt to reproduce those irregular boundary cases.
"""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Return the position (0, 1, or 2) of the median in each row of x."""
    values = np.asarray(x)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("x must have shape (n, 3)")
    return np.argsort(values, axis=1)[:, 1]
