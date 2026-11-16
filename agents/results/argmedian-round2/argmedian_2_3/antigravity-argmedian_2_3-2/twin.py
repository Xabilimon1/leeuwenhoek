"""Twin re-implementation of the 2-hidden-unit median RNN.

The network represents a degenerate heuristic:
- It NEVER predicts position 2 (logit 2 is mathematically shadowed by logits 0 and 1).
- It defaults to predicting position 0.
- It predicts position 1 only when x_2 is sufficiently positive relative to x_1,
  and x_3 lies in an intermediate range relative to the residual of x_2.

Named constants derived from the weights:
- C_REL = 0.870246: recurrent threshold ratio -(w_hh @ w_ih)[1] / w_ih[1].
  Determines when x_2 dominates x_1 (or when x_3 dominates x_2 if x_1 <= 0).
- D_LOW = 1.986058: lower boundary multiplier -w_hh[1, 1] for x_3 to excite unit 1.
- D_HIGH = 20.02711: upper boundary multiplier where unit 0 overtakes unit 1 in readout.
"""

import numpy as np

C_REL = 0.87024635
D_LOW = 1.9860576
D_HIGH = 20.02711


def predict(x: np.ndarray) -> np.ndarray:
    """Predict median positions (n,), ints in 0..2, for inputs x of shape (n, 3)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]

    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]

    preds = np.zeros(x.shape[0], dtype=np.int64)

    # Condition A: x_1 <= 0, x_2 > 0, and x_3 > C_REL * x_2
    mask_a = (x1 <= 0.0) & (x2 > 0.0) & (x3 > C_REL * x2)

    # Condition B: x_1 > 0, x_2 > C_REL * x_1, and x_3 in (D_LOW * delta, D_HIGH * delta)
    delta = x2 - C_REL * x1
    mask_b = (x1 > 0.0) & (delta > 0.0) & (x3 > D_LOW * delta) & (x3 < D_HIGH * delta)

    # The network predicts 1 on mask_a or mask_b, and defaults to 0 elsewhere.
    # Position 2 is never predicted.
    preds[mask_a | mask_b] = 1

    return preds
