"""Where do the 0.064 points between the error law and the trained h=4 net come from?

The law is exact for the ideal designed net. The trained net differs from it in
a few places. Written in the sum/difference basis (design.py), the trained net
(mirror-symmetrized) is the ideal one plus:
  same-half block, comparison row   (p, q)   read with -/+ signs
  same-half block, memory row       (c, e)
  cross-half block, comparison row
  cross-half block, memory row
  readout alphas                    (the betas are read off exactly)
plus the mirror asymmetry (trained vs its symmetrized version).
We transplant each piece alone into the ideal net, then all of them, and
measure the exact error each time.

Usage: uv run python scripts/gap_h4.py
"""

import numpy as np

from leeuwenhoek.design import B, BINV, designed_weights, read_off
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import symmetrize

_, w = load_argmedian(4, 3)
w = {k: v.astype(np.float64) for k, v in w.items()}
p = read_off(w)
ideal = designed_weights(p["a"], p["b"], p["e"], p["B2"], p["B3"], dtype=np.float64)
sym = {k: v.astype(np.float64) for k, v in symmetrize(w).items()}


def to_sd(wt):
    U = wt["w_out"] @ BINV
    return {"ih": B @ wt["w_ih"], "hh": B @ wt["w_hh"] @ BINV, "out": U - U[0]}  # readout relative to x1


def from_sd(sd):
    return {"w_ih": BINV @ sd["ih"], "w_hh": BINV @ sd["hh"] @ B, "w_out": sd["out"] @ B}


I, T = to_sd(ideal), to_sd(sym)
# (rows, cols) of each block in the 4x4 sd matrix, both mirror copies
BLOCKS = {
    "same half, comparison row (p, q)": [(0, slice(0, 2)), (2, slice(2, 4))],
    "same half, memory row (c, e)": [(1, slice(0, 2)), (3, slice(2, 4))],
    "cross half, comparison row (p, q)": [(0, slice(2, 4)), (2, slice(0, 2))],
    "cross half, memory row (c, e)": [(1, slice(2, 4)), (3, slice(0, 2))],
}


def err(wt):
    return 1 - exact_accuracy(wt)[0]


def transplant(names):
    sd = {k: v.copy() for k, v in I.items()}
    for n in names:
        if n == "readout alphas":
            sd["out"][:, [0, 2]] = T["out"][:, [0, 2]]
        else:
            for r, c in BLOCKS[n]:
                sd["hh"][r, c] = T["hh"][r, c]
    return from_sd(sd)


law = np.sqrt(3) / (2 * np.pi) * (p["b"] / (p["e"] * p["a"])) ** 2
e_ideal, e_sym, e_model = err(ideal), err(sym), err(w)
print(f"law {law:.4%}   ideal net (exact) {e_ideal:.4%}   symmetrized trained {e_sym:.4%}   trained {e_model:.4%}")
print(f"gap to explain: trained - law = {e_model - law:+.4%}\n")
print(f"  {'ideal -> law (higher-order terms)':<44} {e_ideal - law:+.4%}")
parts = list(BLOCKS) + ["readout alphas"]
for n in parts:
    print(f"  {'+ ' + n:<44} {err(transplant([n])) - e_ideal:+.4%}")
e_all = err(transplant(parts))
print(f"  {'all transplants together':<44} {e_all - e_ideal:+.4%}   (exact error {e_all:.4%})")
print(f"  {'interaction (together - sum of singles)':<44} "
      f"{(e_all - e_ideal) - sum(err(transplant([n])) - e_ideal for n in parts):+.4%}")
print(f"  {'mirror asymmetry (trained - symmetrized)':<44} {e_model - e_sym:+.4%}")
print(f"  {'check: all transplants == symmetrized?':<44} {abs(e_all - e_sym):.2e}")

# Split by kind of error instead (exact): the ideal net only has the step-3
# x1-by-default band; the trained net adds errors where its comparisons leak.
from collections import defaultdict  # noqa: E402


def kinds(wt):
    _, _, pieces = exact_accuracy(wt)
    out = defaultdict(float)
    for q in pieces:
        if q["pred"] == q["order"][1]:
            continue
        steps = q["path"].split("-")
        mixed = [t for t in (1, 2) if "1" in steps[t][:2] and "1" in steps[t][2:]]
        if steps[2] == "0000" or (mixed == [2] and q["pred"] == 0):
            k = "x2~x3 tie, says x1 (band + dead)"
        elif mixed and mixed[0] == 1:
            k = "x1~x2 tie, says x2 (shifted)" if q["pred"] == 1 else "x1~x2 tie, step 3 then wrong"
        elif mixed:
            k = "x2~x3 tie, other answer"
        else:
            k = "turn: x1 vs x3 boundary shifted"
        out[k] += q["prob"]
    return out


rows = {"ideal": kinds(ideal), "symmetrized": kinds(sym), "trained": kinds(w)}
names = sorted({k for r in rows.values() for k in r}, key=lambda k: -rows["trained"][k])
print(f"\n{'kind of error':<36} {'ideal':>9} {'symmetrized':>12} {'trained':>9} {'trained - ideal':>16}")
for k in names:
    r = {n: rows[n][k] for n in rows}
    print(f"{k:<36} {r['ideal']:9.4%} {r['symmetrized']:12.4%} {r['trained']:9.4%} {r['trained'] - r['ideal']:+16.4%}")
