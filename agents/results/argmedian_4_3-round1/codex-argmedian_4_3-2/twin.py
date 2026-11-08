"""Simple rule-level twin for the three-input recurrent classifier."""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Return the position of the median in each row of an (n, 3) array."""
    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("x must have shape (n, 3)")
    # Stable sorting gives deterministic behavior for exact ties.
    order = np.argsort(values, axis=1, kind="stable")
    return order[:, 1].astype(int)
