"""The weight twin built from the explanation (design.py) against argmedian h=4 L=3.

1. Do the derived constants match the trained weights?  (sum/difference basis)
2. Agreement with the model, entry by entry, and exact accuracy of the twin.
3. Sensitivity to the three free numbers b/a, e, B2/B3.

Usage: uv run python scripts/design_twin_h4.py
"""

import numpy as np

from leeuwenhoek.design import B, BINV, derived, designed_weights
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import symmetrize

np.set_printoptions(precision=3, suppress=True)
_, w = load_argmedian(4, 3)
s = symmetrize(w)
W = B @ s["w_hh"].astype(np.float64) @ BINV
w_ih_sd = B @ s["w_ih"]
U = s["w_out"] @ BINV

# 1. Read a, b, e off the trained net, derive the rest, compare
a, b = w_ih_sd[0], w_ih_sd[1]
e_same, e_cross = -W[1, 1], W[1, 3]
e = (e_same + e_cross) / 2
k = derived(a, b, e)
print(f"trained: a = {a:.3f}, b = {b:.3f}, e = {e_same:.3f} (same half) / {e_cross:.3f} (cross)  -> use e = {e:.3f}")
print(f"{'constant':<10} {'derived':>9} {'trained same':>13} {'trained cross':>14}")
for name, val, same, cross in [
    ("p", k["p"], -W[0, 0], W[0, 2]),
    ("q", k["q"], -W[0, 1], W[0, 3]),
    ("c", k["c"], W[1, 0], -W[1, 2]),
]:
    print(f"{name:<10} {val:9.3f} {same:13.3f} {cross:14.3f}")
rel = U - U[0]  # readout relative to the x1 logit (a common offset changes nothing)
B2, B3 = rel[1, 1], -rel[2, 1]
print(f"readout: beta_x2 = {B2:.2f}, beta_x3 = {-B3:.2f}; alpha_x2 derived {-B2 * b / a:.3f} vs trained {rel[1, 0]:.3f};"
      f" alpha_x3 derived {-B3 * k['g'] / a:.3f} vs trained {rel[2, 0]:.3f}\n")

# 2. Agreement and exact accuracy
x = gaussian_inputs(4_000_000, 3, seed=12)
model = forward(w, x).argmax(-1)
target = argmedian_target(x)
model_acc, _, _ = exact_accuracy(w)


def report(name, wd):
    acc, _, _ = exact_accuracy(wd)
    agree = np.mean(forward(wd, x).argmax(-1) == model)
    print(f"  {name:<44} agreement {agree:8.3%}   exact accuracy {acc:8.4%}")


print(f"model exact accuracy {model_acc:.4%}")
report("designed, constants read off the model", designed_weights(a, b, e, B2, B3))
report("designed, round numbers (a=8.7 b=0.74 e=0.37 B=15.5)", designed_weights(8.7, 0.74, 0.37, 15.5, 15.5))

# 3. Sensitivity
print("\nsensitivity (other numbers as read off the model):")
for bb in [0.3, 0.5, 1.0, 1.5]:
    report(f"b = {bb}", designed_weights(a, bb, e, B2, B3))
for ee in [0.25, 0.5]:
    report(f"e = {ee}", designed_weights(a, b, ee, B2, B3))
for r in [0.5, 2.0]:
    report(f"B2/B3 = {r}", designed_weights(a, b, e, B2 * r, B3))

# 4. Error law: the pair of neurons breaks when |sigma| < |mu|, a band whose width
# grows with b/a. Fit error = K (b/a)^gamma on designed nets (exact accuracies),
# at the model's e, and compare the prediction for the model's own b/a.
bs = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
errs = np.array([1 - exact_accuracy(designed_weights(a, bb, e, B2, B3, dtype=np.float64))[0] for bb in bs])
gamma, logk = np.polyfit(np.log(bs / a), np.log(errs), 1)
print(f"\nerror law on designed nets (e = {e:.3f}): error = {np.exp(logk):.3f} * (b/a)^{gamma:.3f}")
for bb, er in zip(bs, errs):
    print(f"  b/a = {bb / a:.4f}: exact error {er:.5%}  law {np.exp(logk) * (bb / a) ** gamma:.5%}")
pred = np.exp(logk) * (b / a) ** gamma
print(f"law at the model's b/a = {b / a:.4f}: error {pred:.4%}; model's exact error {1 - model_acc:.4%}")
