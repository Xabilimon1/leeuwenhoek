"""Print the weights of an argmedian L=3 model and which ReLU neurons fire at each step.

Usage: uv run python scripts/relu_patterns.py [--hidden 4]
"""

import argparse
from collections import Counter

import numpy as np

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian

parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=4)
parser.add_argument("--n", type=int, default=1_000_000)
args = parser.parse_args()

np.set_printoptions(precision=2, suppress=True)
_, w = load_argmedian(args.hidden, 3)
for name, v in w.items():
    print(f"{name}\n{v}\n")

x = gaussian_inputs(args.n, 3)
logits, states = forward(w, x, return_states=True)
wrong = logits.argmax(-1) != argmedian_target(x)
on = states > 0  # (n, L, h): which neurons are active after each step


def pattern(row):
    return "".join("1" if b else "0" for b in row)


for t in range(3):
    counts = Counter(map(pattern, on[:, t]))
    print(f"step {t + 1}  (pattern = neurons 1..{args.hidden} on/off: freq, error rate)")
    for p, c in counts.most_common():
        m = np.array([pattern(r) == p for r in on[:, t]]) if c < args.n else np.ones(args.n, bool)
        print(f"   {p}: {c / args.n:7.3%}   err {wrong[m].mean():6.2%}")
