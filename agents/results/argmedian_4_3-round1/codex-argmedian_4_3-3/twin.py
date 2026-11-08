"""A compact median-rule approximation to the trained three-input model."""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Return the position of the middle-valued element in each row."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (n, 3)")
    # The stable ordering also gives deterministic behavior for exact ties.
    return np.argsort(x, axis=1, kind="stable")[:, 1]
