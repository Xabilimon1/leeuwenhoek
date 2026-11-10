"""Hand-written twin of the 2-unit ReLU RNN (see EXPLANATION.md).

The network only ever answers position 0 or 1 (never 2). It answers 1 iff
unit 2 ("x2 rose above a damped copy of x1") is still alive at t=3 and
x3 lies in a band relative to that rise; otherwise it answers 0.
"""
import numpy as np

K = 0.8702     # damping: unit 2 at t=2 is ~ x2 - K*x1         (= -w_hh[1,1] - w_hh[1,0]*w_ih[0]/w_ih[1])
C_LO = 1.9861  # x3 must exceed C_LO * rise for unit 2 to survive (= -w_hh[1,1])
C_HI = 20.027  # above C_HI * rise unit 1 wins the readout       (see EXPLANATION.md)
G = 79.169     # unit 1 at t=2 only survives if x2 > G*x1      (= -w_hh[0,0] - w_hh[0,1]*w_ih[1]/w_ih[0])
M = 86.61      # extra inhibition from a small positive x1 that unit 1 still carries (EXPLANATION.md)
D = 2045.0     # unit 1 at t=3 overtakes unit 2 in the readout when x3 > D*x2 (EXPLANATION.md)


def predict(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    # Case A: x1 <= 0 (or x1 tiny compared to x2): both units fire at t=2 on x2,
    # unit 1's input to unit 2 cancels part of its self-inhibition -> x3 > K*x2.
    caseA = (x2 > 0) & ((x1 <= 0) | (x2 > G * x1)) \
        & (x3 > K * x2 + M * np.maximum(x1, 0)) & (x3 < D * x2)
    # Case B: x1 > 0: unit 1 is silenced at t=2, unit 2 holds the rise U = x2 - K*x1.
    U = x2 - K * x1
    caseB = (x1 > 0) & (x2 <= G * x1) & (U > 0) & (x3 > C_LO * U) & (x3 < C_HI * U)
    return (caseA | caseB).astype(np.int64)
