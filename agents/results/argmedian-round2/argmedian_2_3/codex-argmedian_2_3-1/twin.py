"""A small rule-based approximation of the supplied recurrent model."""

import numpy as np


# Derived in EXPLANATION.md from the recurrent and readout weights.
_GAMMA = 0.8702461901547263
_LOWER = 1.9860576
_UPPER = 20.027107212641443


def predict(x: np.ndarray) -> np.ndarray:
    """Predict positions for an array with shape (n, 3), returning 0 or 1.

    The rule uses the positive second-step carry proxy p and answers 1 only
    when x3 lies in the corresponding final-input band.
    """
    x = np.asarray(x, dtype=float)
    positive_first = np.maximum(x[:, 0], 0.0)
    carry = np.maximum(0.0, x[:, 1] - _GAMMA * positive_first)
    in_band = (carry > 0.0) & (x[:, 2] > _LOWER * carry) & (x[:, 2] < _UPPER * carry)
    return in_band.astype(int)
