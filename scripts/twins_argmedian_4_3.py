"""Agreement of the handmade twins with argmedian h=4 L=3, entry by entry.

agreement = fraction of inputs where twin and model give the same answer.
The "vuelta" asks for 100%.

Usage: uv run python scripts/twins_argmedian_4_3.py [--n 1000000]
"""

import argparse

import numpy as np

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import algorithmic_twin, round_weights, symmetrize

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1_000_000)
args = parser.parse_args()

_, w = load_argmedian(4, 3)
x = gaussian_inputs(args.n, 3, seed=1)  # fresh inputs, not the ones we explored with
model = forward(w, x).argmax(-1)
target = argmedian_target(x)
print(f"model accuracy {np.mean(model == target):.3%}\n")


def row(name, pred):
    print(f"  {name:<44} agreement {np.mean(pred == model):8.3%}   accuracy {np.mean(pred == target):8.3%}")


print("Algorithmic twin")
row("clean algorithm (eps3 = 0)", algorithmic_twin(x))
# eps3 read off the mixed@3 path matrix: the x2, x3 logits carry about
# -1.7*|x1| against slopes of about 130 per unit of (x2 - x3): 1.7 / 130 ~ 0.013
row("+ step-3 tie band, eps3 = 0.013 (from weights)", algorithmic_twin(x, 0.013))
for e in [0.005, 0.02, 0.03]:
    row(f"  sensitivity: eps3 = {e}", algorithmic_twin(x, e))

print("\nWeight twins (simplification ladder)")
row("original weights (sanity check)", model)
s = symmetrize(w)
row("exact mirror symmetry", forward(s, x).argmax(-1))
for step in [0.01, 0.1, 0.5, 1.0]:
    row(f"symmetry + round to {step}", forward(round_weights(s, step), x).argmax(-1))

print("\nWhich weights are fragile? (symmetry + round ONE group to 0.1)")
for k in ["w_ih", "w_hh", "w_out"]:
    t = dict(s)
    t[k] = round_weights({k: s[k]}, 0.1)[k]
    row(f"round only {k}", forward(t, x).argmax(-1))
