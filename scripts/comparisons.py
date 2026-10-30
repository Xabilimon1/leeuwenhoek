"""Hypothesis: at each step the active half of the network encodes one comparison.

step 1: x1 > 0      step 2: x2 > x1      step 3: x3 > x2
"Positive half" = neurons 1, 2 (w_ih > 0); "negative half" = neurons 3, 4.

Usage: uv run python scripts/comparisons.py
"""

import numpy as np

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import gaussian_inputs
from leeuwenhoek.models import load_argmedian

_, w = load_argmedian(4, 3)
x = gaussian_inputs(1_000_000, 3)
_, states = forward(w, x, return_states=True)
positive = states[:, :, :2].sum(-1) > states[:, :, 2:].sum(-1)

refs = [("x1 > 0", x[:, 0] > 0), ("x2 > x1", x[:, 1] > x[:, 0]), ("x3 > x2", x[:, 2] > x[:, 1])]
for t, (name, ref) in enumerate(refs):
    print(f"step {t + 1}: positive half dominates == ({name}) in {np.mean(positive[:, t] == ref):.2%} of inputs")
