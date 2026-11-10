"""Simple order-statistic twin for the three-input median network."""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Return the position of the middle value for each row of ``x``."""
    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("x must have shape (n, 3) or (3,)")
    return np.argsort(values, axis=1)[:, 1].astype(int)
