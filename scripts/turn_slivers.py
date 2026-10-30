"""Turn paths of argmedian h=4 L=3: where the model's x1-vs-x3 boundary sits.

When the sequence turns (x2 is the min or the max), the median is decided by
comparing x1 with x3. Inside the clean turn paths the logits are linear, so
the model's boundary L_x1 = L_x3 is a plane; we print it as
    x3 = a * x1 + b * x2      (true boundary: a = 1, b = 0)
These coefficients feed estimate.band_cones ("turn sliver").

Usage: uv run python scripts/turn_slivers.py
"""

import numpy as np

from leeuwenhoek.linear import parse_path, path_matrices
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import symmetrize

np.set_printoptions(precision=4, suppress=True)
_, w = load_argmedian(4, 3)
for label, weights in [("symmetrized", symmetrize(w)), ("original", w)]:
    print(label)
    for path, case in [("1100-0011-1100", "x2 min (down, up)"), ("1100-1100-0011", "x2 max (up, down)")]:
        M, _ = path_matrices(weights, parse_path(path))
        d = M[0] - M[2]  # L_x1 - L_x3 = d . x
        a, b = d[0] / -d[2], d[1] / -d[2]
        side = "x1 wins above" if d[2] > 0 else "x1 wins below"
        print(f"  {case:<18} path {path}: boundary x3 = {a:.4f} x1 {b:+.4f} x2  ({side})")
