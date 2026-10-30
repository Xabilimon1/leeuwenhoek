"""Reproduce the argmedian L=3 accuracy table from docs/plan.md.

Usage: uv run python scripts/accuracy_table.py [--n 1000000] [--seed 0]
"""

import argparse
import math

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import accuracy, argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

x = gaussian_inputs(args.n, 3, seed=args.seed)
targets = argmedian_target(x)

print(f"argmedian L=3, n={args.n:,}, seed={args.seed}")
print(f"{'h':>3} {'params':>7} {'accuracy':>9} {'±1 s.e.':>8}")
for h in [2, 4, 8]:
    _, w = load_argmedian(h, 3)
    params = sum(v.size for v in w.values())
    acc = accuracy(forward(w, x), targets)
    se = math.sqrt(acc * (1 - acc) / args.n)
    print(f"{h:>3} {params:>7} {acc:>9.4%} {se:>8.4%}")
