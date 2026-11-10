"""Hand-written twin of the 2-unit ReLU RNN (see EXPLANATION.md).

Rule: the network answers position 1 only when the sequence keeps rising:
x2 must exceed R_RISE * max(x1, 0), and then x3 must exceed a threshold that
grows with how far x2 rose. Otherwise it answers 0. It never answers 2.
"""
import numpy as np

R_RISE = 0.8702   # x2 must beat 0.87 * x1 (unit b survives step 2)
R_GATE = 79.17    # if x2 > 79.17 * x1, unit a also survives step 2
C_LO = 1.986      # x3 must exceed C_LO * rise (unit b survives step 3)
C_HI = 20.03      # ... but stay below C_HI * rise (else unit a wins readout)
M_X1 = 86.61      # threshold slope on x1 when unit a survived step 2
CAP_X2 = 2045.1   # cap on x3 when unit a survived step 2: x3 < CAP_X2*x2 - CAP_X1*x1
CAP_X1 = 160344.0


def predict(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    p = np.maximum(x1, 0.0)            # memory of step 1 (zero if x1 <= 0)
    rise = x2 - R_RISE * p             # how far x2 rose above 0.87*x1
    a_alive = x2 > R_GATE * p          # unit a active after step 2
    one_quiet = (x3 > C_LO * rise) & (x3 < C_HI * rise)
    one_loud = (x3 > R_RISE * x2 + M_X1 * p) & (x3 < CAP_X2 * x2 - CAP_X1 * p)
    out1 = (rise > 0) & np.where(a_alive, one_loud, one_quiet)
    return out1.astype(int)
