"""A compact median-rule approximation to the trained recurrent model."""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Return the position of the middle value in each length-three row."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (n, 3)")
    ranks = np.argsort(np.argsort(x, axis=1), axis=1)
    return np.argmax(ranks == 1, axis=1)
