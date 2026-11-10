"""Median-position rule for the 3-input ReLU RNN.

The network's readout is a pairwise comparison whose zero sits on the
tie planes x_i = x_j, tilted by an angle of order 10^{-3} (see
EXPLANATION.md). On a continuous input law that tilt is a thin wedge,
so the rule implemented here is the untilted comparison: return the
position of the median.

No weights are read or stored. Ties are measure-zero for the Gaussian
inputs the model was trained on; ``argsort`` breaks them toward the
smaller index.
"""

import numpy as np


def predict(x: np.ndarray) -> np.ndarray:
    """Position of the median of each row.

    x: (n, 3) float -> (n,) ints in {0, 1, 2}.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"expected x of shape (n, 3), got {x.shape}")
    return np.argsort(x, axis=1)[:, 1].astype(np.int64)
