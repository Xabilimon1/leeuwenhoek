"""FLOPs of the mechanistic estimate of argmedian h=4 L=3 versus sampling (check suggested by ARC).

Two comparisons, both against the exact accuracy computed from the weights:
  - FLOPs sampling needs to reach the estimate's error (1 standard error);
  - the error sampling reaches with the estimate's FLOPs.
Sampling cost counts only the model (forward pass, argmax, true median), not the
random numbers, which favours sampling.

Usage: uv run python scripts/jacob_flops.py
"""

import numpy as np

from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.flops import estimate_flops, forward_flops
from leeuwenhoek.models import load_argmedian

_, w = load_argmedian(4, 3)
p, _, _ = exact_accuracy(w)
est, flops, stages = estimate_flops(w)
err = abs(est - p)
per_sample = forward_flops(4, 3)

print(f"exact accuracy {p:.6%}; mechanistic estimate {est:.6%}; error {100 * err:.4f} points\n")
print("estimate FLOPs by stage:")
for k, v in stages.items():
    print(f"  {k:<28} {v:>9,}")
print(f"  {'total':<28} {flops:>9,}\n")

n_match = p * (1 - p) / err**2
print(f"one model evaluation: {per_sample} FLOPs")
print(f"sampling to the estimate's error: {n_match:,.0f} samples = {n_match * per_sample:,.0f} FLOPs"
      f"  -> estimate is {n_match * per_sample / flops:.1f}x cheaper")
n_same = flops / per_sample
se = np.sqrt(p * (1 - p) / n_same)
print(f"sampling with the estimate's FLOPs: {n_same:,.0f} samples, error {100 * se:.3f} points"
      f"  ({se / err:.1f}x the estimate's error)")
