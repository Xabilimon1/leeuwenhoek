"""What does each half's 2-neuron state store? Rewrite it as sum and difference.

Mirror pairing: positive half (n1, n2) <-> negative half (n4, n3).
    s = n1 + n2 (or n4 + n3)      d = n2 - n1 (or n3 - n4)
Inside a clean path every state is a linear form in (x1, x2, x3); we print
the coefficients of s and d at each step for the main paths, plus the
recurrent weights rewritten in this basis.

Usage: uv run python scripts/memory_basis.py
"""

import numpy as np

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import symmetrize

np.set_printoptions(precision=2, suppress=True)
_, w = load_argmedian(4, 3)
w = {k: v.astype(np.float64) for k, v in symmetrize(w).items()}  # exact mirror

# Basis change: rows give (s_pos, d_pos, s_neg, d_neg) from (n1, n2, n3, n4)
B = np.array([[1, 1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]], float)
Binv = np.linalg.inv(B)
print("w_ih in (s+, d+, s-, d-):", B @ w["w_ih"])
print("w_hh in (s+, d+, s-, d-) basis (row = target, col = source):")
print(B @ w["w_hh"] @ Binv)
print("w_out in that basis (rows = logits x1 x2 x3):")
print(w["w_out"] @ Binv)

# Linear forms along clean paths (x1 > 0 by symmetry): which half is on per step
for path in ["+-+", "+--", "++-", "+++", "+-"]:
    h = np.zeros((4, 3))
    print(f"\npath {path} (half on at each step):")
    for t, half in enumerate(path):
        z = w["w_hh"] @ h
        z[:, t] += w["w_ih"]
        mask = np.array([1, 1, 0, 0] if half == "+" else [0, 0, 1, 1], float)
        h = mask[:, None] * z
        sd = B @ h
        idx = (0, 1) if half == "+" else (2, 3)
        print(f"  step {t + 1}: s = {sd[idx[0]]} . x    d = {sd[idx[1]]} . x")

# Fragility: round W_hh to 0.1 in the (s, d) basis, rows of one channel at a time
x = gaussian_inputs(1_000_000, 3, seed=1)
ref = forward(w, x).argmax(-1)  # symmetrized model (99.79% = original)
W_sd = B @ w["w_hh"] @ Binv
print("\nagreement after rounding W_hh rows (s/d basis) to 0.1:")
for name, rows in [("s rows (0, 2)", [0, 2]), ("d rows (1, 3)", [1, 3]), ("all rows", [0, 1, 2, 3])]:
    Wr = W_sd.copy()
    Wr[rows] = np.round(Wr[rows], 1)
    v = dict(w, w_hh=Binv @ Wr @ B)
    print(f"  {name:<14} {np.mean(forward(v, x).argmax(-1) == ref):.3%}")
