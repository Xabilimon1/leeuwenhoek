"""Hand-written twin of the 2-unit ReLU RNN (see EXPLANATION.md).

The network only ever answers position 0 or 1.  It answers 1 exactly when
x2 "rises" above (a fraction of) the positive part of x1 and x3 then lands in a
window above that rise; otherwise it answers 0 (also the default when all
logits are 0).
"""
import numpy as np

# Constants derived in EXPLANATION.md
K = 0.8702    # step-2 inhibition of x2 by relu(x1): -(w_hh[1] . w_ih) / w_ih[1]
GATE = 79.17  # unit a survives step 2 only if x2 > GATE * relu(x1): -(w_hh[0] . w_ih) / w_ih[0]
LO = 1.986    # x1>0 branch: x3 must exceed LO * rise  (= -w_hh[1,1])
HI = 20.03    # x1>0 branch: x3 must stay below HI * rise (readout boundary b/a = 1.111)
LO0 = 0.8702  # x1<=0 branch ("reset"): x3 must exceed LO0 * x2 (same formula as K)
HI0 = 2045.0  # x1<=0 branch upper edge (practically infinite)


def predict(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    p1 = np.maximum(x1, 0.0)
    reset = x2 > GATE * p1          # x1 <= 0 (and x2 > 0), or x2 huge vs x1
    rise = np.maximum(x2 - K * p1, 0.0)
    # "reset" branch: step 2 looks like a fresh step 1 with input x2
    one_reset = reset & (x2 > 0) & (x3 > LO0 * x2) & (x3 < HI0 * x2)
    # normal branch: only unit b carries the rise into step 3
    one_norm = ~reset & (rise > 0) & (x3 > LO * rise) & (x3 < HI * rise)
    return np.where(one_reset | one_norm, 1, 0).astype(int)
