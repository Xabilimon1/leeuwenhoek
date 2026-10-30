"""Exact error budget of argmedian h=4 L=3, and how much of it our explanation covers.

Every number here is an exact cone probability (no sampling):
  - error per path category (same categories as error_budget.py)
  - overlap of each error piece with the explained cones (estimate.explained_cones):
    recall = explained errors / all errors, precision = errors / band area.

Usage: uv run python scripts/exact_error_budget.py
"""

from collections import defaultdict

import numpy as np

from leeuwenhoek.cones import cone_fraction
from leeuwenhoek.estimate import explained_cones
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.models import load_argmedian

_, w = load_argmedian(4, 3)
acc, _, pieces = exact_accuracy(w)
err = 1 - acc
print(f"exact accuracy {acc:.5%}, error {err:.5%}\n")


def category(p):
    steps = p["path"].split("-")
    if steps[2] == "0000":
        return "dead"
    for t in (1, 2):
        s = steps[t]
        if "1" in s[:2] and "1" in s[2:]:
            return f"mixed@{t + 1}"
    if p["order"] in [(0, 1, 2), (2, 1, 0)]:
        return "clean-monotone"
    return "clean-turn"


by_cat = defaultdict(lambda: [0.0, 0.0])
for p in pieces:
    c = by_cat[category(p)]
    c[0] += p["prob"]
    c[1] += p["prob"] * (p["pred"] != p["order"][1])
print(f"{'category':<15} {'inputs':>9} {'error':>9} {'share of errors':>16}")
for c in ["clean-monotone", "clean-turn", "mixed@2", "mixed@3", "dead"]:
    mass, e = by_cat[c]
    print(f"{c:<15} {mass:9.4%} {e / mass:9.3%} {e / err:16.1%}")

# Mirror each band to x1 < 0 as well (x -> -x flips every normal)
bands = {}
for name, (normals, _) in explained_cones().items():
    bands[name + " (x1>0)"] = np.asarray(normals, float)
    bands[name + " (x1<0)"] = -np.asarray(normals, float)

print("\nExplained bands vs the model's errors (exact):")
tot_hit = tot_area = 0.0
for name, b in bands.items():
    area = cone_fraction(b)
    hit = sum(cone_fraction(list(p["normals"]) + list(b)) for p in pieces if p["pred"] != p["order"][1])
    tot_hit += hit
    tot_area += area
    print(f"  {name:<36} area {area:.4%}  model errors inside {hit:.4%}  precision {hit / area:6.1%}")
print(f"  {'all bands':<36} area {tot_area:.4%}  model errors inside {tot_hit:.4%}  precision {tot_hit / tot_area:6.1%}")
print(f"\nrecall: the bands contain {tot_hit / err:.1%} of the model's errors")
print(f"unexplained error mass: {err - tot_hit:.4%}")

# Largest unexplained error pieces (error mass not inside any band)
names = ["x1", "x2", "x3"]
rows = []
for p in pieces:
    if p["pred"] == p["order"][1]:
        continue
    inside = sum(cone_fraction(list(p["normals"]) + list(b)) for b in bands.values())
    rows.append((p["prob"] - inside, p))
rows.sort(key=lambda r: -r[0])
print("\nLargest unexplained error pieces:")
for rest, p in rows[:14]:
    o = "<".join(names[i] for i in p["order"])
    print(f"  {rest:.4%}  {category(p):<14} path {p['path']}  order {o}  says {names[p['pred']]}")
