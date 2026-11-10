"""A threshold-rule approximation of the three-step median network."""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Predict positions for an (n, 3) array using a short threshold rule."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (n, 3)")

    first, second, third = x.T
    k = 0.870246  # cancellation in the second hidden unit
    # The first unit's positive carry is normally wiped out by its -76.49
    # self-connection. When x1 <= 0, x2 first creates a two-unit state; when
    # x1 > 0, the second unit carries only the residual above k*x1.
    residual = np.where(
        first <= 0.0,
        np.maximum(second, 0.0),
        np.maximum(second - k * first, 0.0),
    )
    threshold = np.where(first <= 0.0, k * residual, 1.986058 * residual)
    choose_one = (residual > 0.0) & (third > threshold)
    return choose_one.astype(np.int64)
