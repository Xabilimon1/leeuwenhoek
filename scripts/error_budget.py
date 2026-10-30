"""Error budget of argmedian h=4 L=3: where do the errors come from?

Each input is classified by its activation path:
  clean-monotone  every step one half only (1100/0011); comparisons say up-up or down-down
  clean-turn      every step one half only; comparisons turn (x2 is min or max)
  mixed@t         at step t one neuron from each half is on (a near-tie at that step)
  dead            final state all zero (logits all 0, argmax falls back to x1)
  other           anything else

Usage: uv run python scripts/error_budget.py [--n 1000000]
"""

import argparse

import numpy as np

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1_000_000)
args = parser.parse_args()

_, w = load_argmedian(4, 3)
x = gaussian_inputs(args.n, 3)
logits, states = forward(w, x, return_states=True)
pred = logits.argmax(-1)
wrong = pred != argmedian_target(x)

on = states > 0
pos, neg = on[:, :, :2].any(-1), on[:, :, 2:].any(-1)
clean = pos ^ neg  # exactly one half active at each step
mixed = pos & neg
up2, up3 = x[:, 1] > x[:, 0], x[:, 2] > x[:, 1]

cat = np.full(args.n, "other", dtype=object)
cat[clean.all(-1) & (up2 == up3)] = "clean-monotone"
cat[clean.all(-1) & (up2 != up3)] = "clean-turn"
for t in (2, 1):  # earlier step wins if both are mixed
    cat[mixed[:, t]] = f"mixed@{t + 1}"
cat[~on[:, 2].any(-1)] = "dead"

print(f"total error {wrong.mean():.3%}  ({wrong.sum()} of {args.n})\n")
print(f"{'category':<15} {'inputs':>8} {'error':>8} {'share of errors':>16} {'wrong answer = x1':>18}")
for c in ["clean-monotone", "clean-turn", "mixed@2", "mixed@3", "dead", "other"]:
    m = cat == c
    if not m.any():
        continue
    e = wrong & m
    x1 = (pred[e] == 0).mean() if e.any() else float("nan")
    print(f"{c:<15} {m.mean():8.2%} {wrong[m].mean():8.2%} {e.sum() / wrong.sum():16.1%} {x1:18.1%}")

# Is mixed@t a near-tie between x_t and x_{t-1}? Compare the gap to the input size.
gap = np.abs(np.diff(x, axis=1)) / np.linalg.norm(x, axis=1, keepdims=True)
print("\nmedian |x_t - x_(t-1)| / |x|:")
for t in (2, 3):
    m = cat == f"mixed@{t}"
    print(f"  step {t}: inside mixed@{t} {np.median(gap[m, t - 2]):.3f}   all inputs {np.median(gap[:, t - 2]):.3f}")

# Failure inside mixed@3: do the two candidate logits both drop below the x1 logit?
m = (cat == "mixed@3") & wrong
print(f"\nmixed@3 errors where logit x1 beats both x2 and x3: "
      f"{np.mean(logits[m, 0] > logits[m, 1:].max(-1)):.1%}")
