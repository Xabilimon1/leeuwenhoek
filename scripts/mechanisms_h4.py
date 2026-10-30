"""Score the mechanism list of argmedian h=4 L=3 (mechanisms.py) exactly and by sampling.

  - twin agreement with the model (sampling, fresh inputs)
  - estimate: exact accuracy implied by the cones, vs the exact model accuracy
  - exact coverage: model errors inside the cones / all model errors
  - the largest model error pieces left outside every cone

Usage: uv run python scripts/mechanisms_h4.py
"""

import numpy as np

from leeuwenhoek.cones import cone_fraction
from leeuwenhoek.estimate import error_of_cone
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.forward import forward
from leeuwenhoek.mechanisms import mechanism_cones
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import twin_from_cones

_, w = load_argmedian(4, 3)
cones = mechanism_cones(w)
acc, _, pieces = exact_accuracy(w)
print(f"{len(cones)} cones; exact model accuracy {acc:.5%}\n")

est = 1 - sum(error_of_cone(n, s) for n, s in cones.values())
print(f"estimate from cones: {est:.5%}   error {abs(est - acc):.4%} points")

x = gaussian_inputs(4_000_000, 3, seed=9)
model = forward(w, x).argmax(-1)
twin = twin_from_cones(x, cones)
print(f"twin agreement (4e6 fresh inputs): {np.mean(twin == model):.4%}   twin accuracy {np.mean(twin == argmedian_target(x)):.4%}")

err_pieces = [p for p in pieces if p["pred"] != p["order"][1]]
err = sum(p["prob"] for p in err_pieces)
inside = []
for p in err_pieces:
    hit = sum(cone_fraction(list(p["normals"]) + list(n)) for n, _ in cones.values())
    inside.append(min(hit, p["prob"]))
print(f"exact coverage: {sum(inside) / err:.1%} of the model's error mass")

names = ["x1", "x2", "x3"]
rest = sorted(zip([p["prob"] - h for p, h in zip(err_pieces, inside)], err_pieces), key=lambda r: -r[0])
print(f"left outside: {sum(r for r, _ in rest):.4%} in {sum(r > 1e-6 for r, _ in rest)} pieces; largest:")
for r, p in rest[:10]:
    o = "<".join(names[i] for i in p["order"])
    print(f"  {r:.4%}  path {p['path']}  order {o}  says {names[p['pred']]}")

# Beyond the mechanisms: add the leftover error pieces one by one (largest first)
# as exact path-level cones. This is enumeration, not explanation; the curve shows
# how many extra cones each step of agreement costs.
print("\nagreement vs number of extra path-level cones:")
extra = dict(cones)
done = 0
for k in [0, 10, 30, len(rest)]:
    for r, p in rest[done:k]:
        extra[f"piece {p['path']} {p['order']} says {p['pred']}"] = (p["normals"], p["pred"])
    done = max(done, k)
    t = twin_from_cones(x, extra)
    print(f"  +{k:3d} cones ({len(extra)} total): agreement {np.mean(t == model):.4%}")

# Close the loop: where a mechanism cone claims an error but the model is right
# (false positives), add the exact correct piece on top.
fixes = 0
for p in pieces:
    if p["pred"] != p["order"][1]:
        continue
    if any(cone_fraction(list(p["normals"]) + list(n)) > 0 for n, _ in cones.values()):
        extra[f"correct {p['path']} {p['order']}"] = (p["normals"], p["pred"])
        fixes += 1
t = twin_from_cones(x, extra)
print(f"  +{fixes:3d} correct pieces ({len(extra)} total): agreement {np.mean(t == model):.5%}"
      f"  ({np.sum(t != model)} of {len(x):,} inputs differ)")
w64 = {k: v.astype(np.float64) for k, v in w.items()}
model64 = forward(w64, x.astype(np.float64)).argmax(-1)
print(f"  same twin vs the model run in float64: {np.sum(t != model64)} inputs differ "
      f"(float32 vs float64 model: {np.sum(model != model64)} differ)")
