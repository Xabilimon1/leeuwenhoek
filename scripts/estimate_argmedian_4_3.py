"""Mechanistic accuracy estimate of argmedian h=4 L=3 versus sampling.

The estimate never runs the model: it adds up the exact Gaussian probability
of the error cones we explained (estimate.band_cones).

Usage: uv run python scripts/estimate_argmedian_4_3.py
"""

import numpy as np

from leeuwenhoek.estimate import estimate_accuracy
from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import algorithmic_twin_bands

est, parts = estimate_accuracy()
print("Mechanistic estimate (no model evaluations):")
for k, p in parts.items():
    print(f"  error from {k:<26} {p:.4%}")
print(f"  estimated accuracy {est:.4%}\n")

# Ground truth with a lot of sampling (only to score the estimate)
_, w = load_argmedian(4, 3)
x = gaussian_inputs(10_000_000, 3, seed=4)
target = argmedian_target(x)
true_acc = np.mean(forward(w, x).argmax(-1) == target)
twin_acc = np.mean(algorithmic_twin_bands(x) == target)
se = np.sqrt(true_acc * (1 - true_acc) / len(x))
print(f"model accuracy, 10^7 samples: {true_acc:.4%} (+-{se:.4%})")
print(f"twin accuracy (sanity: should equal the estimate): {twin_acc:.4%}")

err = abs(est - true_acc)
p = true_acc
n_equiv = p * (1 - p) / err**2
print(f"\nestimate error: {err:.4%}")
print(f"sampling reaches that error (1 s.e.) with about {n_equiv:,.0f} model evaluations")
