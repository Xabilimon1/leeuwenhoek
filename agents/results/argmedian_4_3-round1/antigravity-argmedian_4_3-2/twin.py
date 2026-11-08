"""Twin model replicating the mechanistic behavior of the median-finding RNN.

This module implements the behavioral rule derived in EXPLANATION.md:
1. Base prediction: the position (0..2) of the median element of the 3 inputs.
2. Error bands:
   - When x1 and x2 form a tight cluster relative to the outlier x0, the step-3
     recurrent signal is too weak to overcome the default readout, causing
     the network to output position 0 instead of the true median (1 or 2).
   - When x0 is the median but x2 or x1 is extremely close to x0, slight readout
     asymmetries favor positions 2 or 1 over position 0.
"""

import numpy as np

# Named constants derived analytically from the weights in EXPLANATION.md:
# Cluster ambiguity thresholds: relative gap (x2 - x1) / |x0 - (x1+x2)/2|
G_CLUSTER_LEFT = 0.0136
G_CLUSTER_RIGHT = 0.0107

# Boundary overshoot thresholds for position 2 and position 1 when x0 is median
G_02 = 0.0075
G_01 = 0.0040


def predict(x: np.ndarray) -> np.ndarray:
    """Predict position of median (0..2) for input array x of shape (n, 3) or (3,)."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = False
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True

    # Base rule: position of the median element
    true_median = np.argsort(x, axis=1)[:, 1]
    pred = true_median.copy()

    x0 = x[:, 0]
    x1 = x[:, 1]
    x2 = x[:, 2]

    # Error Region 1 (Cluster Ambiguity):
    # When x1 and x2 are very close together relative to their distance from x0,
    # and x0 is an extreme value (outlier), the recurrent signal at step 3 is too weak
    # to overcome the default logit 0, so the network incorrectly predicts position 0.
    is_outlier = (true_median != 0)
    scale_0 = np.abs(x0 - 0.5 * (x1 + x2))

    cond_cluster = is_outlier & (
        ((x0 > x1) & (x2 - x1 >= -G_CLUSTER_LEFT * scale_0) & (x2 - x1 <= G_CLUSTER_RIGHT * scale_0)) |
        ((x0 < x1) & (x2 - x1 >= -G_CLUSTER_RIGHT * scale_0) & (x2 - x1 <= G_CLUSTER_LEFT * scale_0))
    )
    pred[cond_cluster] = 0

    # Error Region 2 (Boundary overshoot for position 2):
    # When x0 is the true median but x2 is extremely close to x0, logit 2 slightly beats logit 0.
    scale_1 = np.abs(x1 - 0.5 * (x0 + x2))
    cond_02 = (true_median == 0) & (np.abs(x2 - x0) <= G_02 * scale_1)
    pred[cond_02] = 2

    # Error Region 3 (Boundary overshoot for position 1):
    # When x0 is the true median but x1 is extremely close to x0, logit 1 slightly beats logit 0.
    scale_2 = np.abs(x2 - 0.5 * (x0 + x1))
    cond_01 = (true_median == 0) & (np.abs(x1 - x0) <= G_01 * scale_2)
    pred[cond_01] = 1

    if squeeze:
        return pred[0]
    return pred
