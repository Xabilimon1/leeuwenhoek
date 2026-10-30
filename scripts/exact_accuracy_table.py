"""Exact accuracy of argmedian L=3 models from their weights (no sampling).

Usage: uv run python scripts/exact_accuracy_table.py
"""

import time

from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.models import load_argmedian

print(f"{'h':>3} {'params':>7} {'exact accuracy':>15} {'paths':>6} {'pieces':>7} {'seconds':>8}")
for h in [2, 4, 8]:
    _, w = load_argmedian(h, 3)
    t0 = time.perf_counter()
    acc, total, pieces = exact_accuracy(w)
    dt = time.perf_counter() - t0
    assert abs(total - 1) < 1e-6, total
    params = sum(v.size for v in w.values())
    paths = len({p["path"] for p in pieces})
    print(f"{h:>3} {params:>7} {acc:>15.6%} {paths:>6} {len(pieces):>7} {dt:>8.2f}")
