"""Hand-written twin of the 4-unit ReLU RNN trained to find the median position.

Rule (see EXPLANATION.md):
  * Default: answer the true median position.
  * Near-tie of x2 and x3 (closer to each other than to x1): the readout cannot
    decide between positions 1 and 2 and falls into the "0" zone, when
    the relative gap (x3-x2)*sign(x2-x1)/|x2-x1| lies in [G23_LO, G23_HI].
  * Near-tie of x1 and x3 (x2 extreme), x1 slightly inside: answers 2
    when (x3-x1)*sign(x1-x2)/|x1-x2| lies in [0, G13].
  * Near-tie of x1 and x2 (x3 extreme), x1 slightly inside: answers 1
    when (x2-x1)*sign(x1-x3)/|x1-x3| lies in [0, G12].
  * Dead state: a same-sign sequence that shrinks slowly towards 0 switches every
    unit off at the last step; all logits are 0 and argmax gives 0.
"""
import numpy as np

G23_LO, G23_HI = -0.018, 0.016
G13 = 0.0063
G12 = 0.0018
DEAD_R21 = (0.53, 1.0)    # range of x2/x1
DEAD_R32 = (0.88, 1.08)   # range of x3/x2


def _rel(xi, xj, xk):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sign(xi - xk) * (xj - xi) / np.abs(xi - xk)


def predict(x):
    x = np.asarray(x, dtype=float)
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    out = np.argsort(x, axis=1)[:, 1].copy()

    d12, d13, d23 = np.abs(x1 - x2), np.abs(x1 - x3), np.abs(x2 - x3)
    # x2 ~ x3
    r = _rel(x2, x3, x1)
    m = (d23 < np.minimum(d12, d13)) & (r >= G23_LO) & (r <= G23_HI)
    out[m] = 0
    # x1 ~ x3
    r = _rel(x1, x3, x2)
    m = (d13 < np.minimum(d12, d23)) & (r >= 0) & (r <= G13)
    out[m] = 2
    # x1 ~ x2
    r = _rel(x1, x2, x3)
    m = (d12 < np.minimum(d13, d23)) & (r >= 0) & (r <= G12)
    out[m] = 1
    # dead state
    with np.errstate(divide="ignore", invalid="ignore"):
        r21, r32 = x2 / x1, x3 / x2
    m = (r21 > DEAD_R21[0]) & (r21 < DEAD_R21[1]) & (r32 > DEAD_R32[0]) & (r32 < DEAD_R32[1])
    out[m] = 0
    return out
