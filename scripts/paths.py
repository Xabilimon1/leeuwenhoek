"""Activation paths of argmedian L=3 models and the linear map inside each path.

A path is the sequence of ReLU patterns over the 3 steps. Inside one path the
network is linear: logits = M @ x for a fixed 3x3 matrix M, and the path is the
set of x where every pre-activation has the sign the pattern says.

Usage: uv run python scripts/paths.py [--hidden 4] [--top 8]
"""

import argparse
from collections import Counter

import numpy as np

from leeuwenhoek.forward import forward
from leeuwenhoek.linear import parse_path, path_matrices
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian

parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=4)
parser.add_argument("--n", type=int, default=1_000_000)
parser.add_argument("--top", type=int, default=8)
parser.add_argument("--path", default=None, help="e.g. 1100-0011-1010; default: most common")
args = parser.parse_args()
np.set_printoptions(precision=2, suppress=True)


_, w = load_argmedian(args.hidden, 3)
x = gaussian_inputs(args.n, 3)
logits, states = forward(w, x, return_states=True)
wrong = logits.argmax(-1) != argmedian_target(x)
keys = ["-".join("".join("1" if b else "0" for b in s) for s in row) for row in states > 0]
counts = Counter(keys)
keys = np.array(keys)

print(f"{len(counts)} distinct paths. Top {args.top}:")
cum = 0
for k, c in counts.most_common(args.top):
    m = keys == k
    cum += c
    print(f"  {k}: {c / args.n:6.2%}  err {wrong[m].mean():6.2%}  (cumulative {cum / args.n:6.2%})")

top = args.path or counts.most_common(1)[0][0]
masks = parse_path(top)
M, pre = path_matrices(w, masks)
m = keys == top
assert np.allclose(x[m] @ M.T, logits[m], atol=1e-3), "path matrix does not reproduce logits"
print(f"\nPath {top} ({m.mean():.2%} of inputs): logits = M @ (x1, x2, x3)")
print(M)
for t, z in enumerate(pre):
    print(f"pre-activations step {t + 1} (rows = neurons, cols = coeff of x1 x2 x3):\n{z}")

# Which true orderings land in this path, and how often each is wrong
order = np.argsort(x[m], axis=-1)
names = np.array(["<".join(f"x{i + 1}" for i in o) for o in order])
pred = logits[m].argmax(-1)
print("\ntrue ordering inside this path:")
for name, c in Counter(names).most_common():
    k = names == name
    wrong_k = wrong[m][k]
    said = np.bincount(pred[k][wrong_k], minlength=3)
    print(f"  {name}: {c / m.sum():6.1%} of path  err {wrong_k.mean():6.2%}  wrong answers x1/x2/x3: {said}")
