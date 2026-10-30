"""Exact error pieces of a designed net: which tie and which wrong answer make up the law.

Usage: uv run python scripts/law_pieces.py [--b 0.05]
"""

import argparse
from collections import defaultdict

import numpy as np

from leeuwenhoek.design import designed_weights, read_off
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.models import load_argmedian

parser = argparse.ArgumentParser()
parser.add_argument("--b", type=float, default=0.05)
args = parser.parse_args()
_, w = load_argmedian(4, 3)
p = read_off(w)
H2 = (args.b / (p["e"] * p["a"])) ** 2
acc, _, pieces = exact_accuracy(designed_weights(p["a"], args.b, p["e"], p["B2"], p["B3"], dtype=np.float64))
names = ["x1", "x2", "x3"]
groups = defaultdict(float)
for q in pieces:
    if q["pred"] == q["order"][1]:
        continue
    o = q["order"]
    key = f"order {'<'.join(names[i] for i in o)}, says {names[q['pred']]}, path {q['path']}"
    groups[key] += q["prob"]
err = 1 - acc
print(f"b = {args.b}: error {err:.5%} = {err / H2:.4f} (b/(e a))^2")
for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"  {v / H2:7.4f} (b/(e a))^2  {k}")
