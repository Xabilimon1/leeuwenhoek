"""A compact, order-based approximation of the trained recurrent model."""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Return the position of the middle-valued entry in each row of x.

    The trained model gets the ordering right on the large majority of inputs.
    Its remaining errors concentrate near close-value comparisons; this small
    rule captures its dominant behavior without copying its recurrent math.
    """
    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("x must have shape (n, 3)")
    return np.argsort(values, axis=1, kind="stable")[:, 1].astype(int)
