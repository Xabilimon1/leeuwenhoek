"""Hand-written re-implementation of the trained median RNN.
Built from the mechanistic explanation of its decision boundaries and tilt imperfections.
"""

import numpy as np

# Named constants derived from the network's boundary hyperplanes in EXPLANATION.md:
# Each constant represents the relative gap threshold between the two competing coordinates
# and the distance to the third coordinate below which the model's tilted boundary misclassifies.
GAP_02 = 0.000982      # Tilt in (1, 0, 2): model predicts 2 instead of 0 when (x2 - x0) < GAP_02 * (x0 - x1)
GAP_21 = 0.000930      # Tilt in (0, 2, 1): model predicts 1 instead of 2 when (x1 - x2) < GAP_21 * (x2 - x0)
GAP_12 = 0.000431      # Tilt in (2, 1, 0): model predicts 2 instead of 1 when (x1 - x2) < GAP_12 * (x0 - x1)
GAP_02_REV = 0.000131  # Tilt in (2, 0, 1): model predicts 2 instead of 0 when (x0 - x2) < GAP_02_REV * (x1 - x0)


def predict(x: np.ndarray) -> np.ndarray:
    """Predict the argmedian position (0..2) for inputs x of shape (n, 3)."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = False
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True

    # Order of inputs: order[:, 0] = min, order[:, 1] = median, order[:, 2] = max
    order = np.argsort(x, axis=1)
    min_idx = order[:, 0]
    med_idx = order[:, 1]
    max_idx = order[:, 2]

    # Baseline rule: output the true median position
    preds = med_idx.astype(np.int64).copy()

    # Apply the systematic boundary tilts discovered in the trained network:

    # 1. Ordering (1, 0, 2): x1 < x0 < x2 -> true median is 0, but when x2 is very close to x0,
    # the readout favors 2 over 0.
    m102 = (min_idx == 1) & (med_idx == 0) & (max_idx == 2)
    if np.any(m102):
        gap = x[m102, 2] - x[m102, 0]
        ref = x[m102, 0] - x[m102, 1]
        err = m102.copy()
        err[m102] = (gap < GAP_02 * ref)
        preds[err] = 2

    # 2. Ordering (0, 2, 1): x0 < x2 < x1 -> true median is 2, but when x1 is very close to x2,
    # the readout favors 1 over 2.
    m021 = (min_idx == 0) & (med_idx == 2) & (max_idx == 1)
    if np.any(m021):
        gap = x[m021, 1] - x[m021, 2]
        ref = x[m021, 2] - x[m021, 0]
        err = m021.copy()
        err[m021] = (gap < GAP_21 * ref)
        preds[err] = 1

    # 3. Ordering (2, 1, 0): x2 < x1 < x0 -> true median is 1, but when x1 is very close to x2,
    # the readout favors 2 over 1.
    m210 = (min_idx == 2) & (med_idx == 1) & (max_idx == 0)
    if np.any(m210):
        gap = x[m210, 1] - x[m210, 2]
        ref = x[m210, 0] - x[m210, 1]
        err = m210.copy()
        err[m210] = (gap < GAP_12 * ref)
        preds[err] = 2

    # 4. Ordering (2, 0, 1): x2 < x0 < x1 -> true median is 0, but when x0 is very close to x2,
    # the readout favors 2 over 0.
    m201 = (min_idx == 2) & (med_idx == 0) & (max_idx == 1)
    if np.any(m201):
        gap = x[m201, 0] - x[m201, 2]
        ref = x[m201, 1] - x[m201, 0]
        err = m201.copy()
        err[m201] = (gap < GAP_02_REV * ref)
        preds[err] = 2

    if squeeze:
        return preds[0]
    return preds
