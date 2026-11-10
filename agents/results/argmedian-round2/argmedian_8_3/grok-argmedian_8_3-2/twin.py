"""Median-position rule for the 3-input network.

The network returns the index of the median input, except on thin bands
along the planes where two inputs are equal. Those bands are described in
EXPLANATION.md. This twin is the median rule itself and does not try to
reproduce the bands.
"""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Position of the median of each row.

    x: array of shape (n, 3).
    returns: int array of shape (n,) with values in {0, 1, 2}.
    """
    x = np.asarray(x, dtype=np.float64)
    # mergesort is stable, so exact ties (probability zero for Gaussian
    # inputs) would break toward the smaller index.
    order = np.argsort(x, axis=1, kind="mergesort")
    return order[:, 1].astype(np.int64)
