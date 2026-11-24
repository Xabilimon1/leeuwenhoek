"""Paired test: does a sign-mirror symmetric initialisation steer argmedian h=4 L=3 to the seed-0 mechanism?

Each seed 25-48 is trained twice with AlgZoo's full recipe: the normal init and
the same init made exactly mirror symmetric (models.mirror_init_, first half kept,
same data stream). Arms live in data/trained/argmedian_4_3_seed<s>/ and
argmedian_4_3_mirror_seed<s>/ (scripts/train_batch_h4.py --arms normal mirror).

Criteria fixed on 2026-11-24 before training (notes/argmedian_4_3_seeds.md):
  mechanism = pilot criterion (mirror residual < 0.1 and >= 80 % of errors at
  near ties) AND same answer as ARC seed 0 on >= 99 % of inputs (best of x and
  -x). The agreement test carries the weight in the mirror arm, where a small
  residual partly comes from the start.
  The idea survives only if: mirror arm >= 10/24, exact one-sided McNemar
  p < 0.05 on the pairs, and not killed by the agreement test (passers must reach
  99 %). Otherwise it is dropped.

Usage: PYTHONPATH=src python scripts/mirror_init_h4.py [--first 25 --last 48] [--n 1000000]
"""

import argparse
from math import comb
from pathlib import Path

import numpy as np
import torch as th
from alg_zoo.architectures import DistRNN

from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian, numpy_weights
from leeuwenhoek.structure import mirror_permutation

MIRROR_MAX, NEAR_TIE_MIN, GAP, AGREE_MIN = 0.1, 0.8, 0.05, 0.99
MIN_MIRROR_ARM, ALPHA = 10, 0.05
ROOT = Path(__file__).resolve().parents[1] / "data" / "trained"

parser = argparse.ArgumentParser()
parser.add_argument("--first", type=int, default=25)
parser.add_argument("--last", type=int, default=48)
parser.add_argument("--n", type=int, default=1_000_000)
args = parser.parse_args()

x = gaussian_inputs(args.n, 3, seed=1)
target = argmedian_target(x)
gap = np.diff(np.sort(x, -1), axis=-1).min(-1) / np.abs(x).max(-1)  # same as seeds_h4.py / pilot
w0 = load_argmedian(4, 3, seed=0)[1]
a0, a0_neg = forward(w0, x).argmax(-1), forward(w0, -x).argmax(-1)


def load(seed, arm, n=2**30):
    tag = "mirror_" if arm == "mirror" else ""
    p = ROOT / f"argmedian_4_3_{tag}seed{seed}" / f"argmedian_4_3_{n}_{seed}.pth"
    if not p.exists():
        return None
    m = DistRNN(hidden_size=4, seq_len=3)
    m.load_state_dict(th.load(p, weights_only=True))
    return numpy_weights(m)


def measure(w):
    pred = forward(w, x).argmax(-1)
    wrong = pred != target
    near = float(np.mean(gap[wrong] < GAP)) if wrong.any() else float("nan")
    res = mirror_permutation(w)[1]
    agree = max(float(np.mean(pred == a0)), float(np.mean(pred == a0_neg)))
    pilot = res < MIRROR_MAX and near >= NEAR_TIE_MIN
    return dict(acc=exact_accuracy(w)[0], res=res, near=near, agree=agree, pilot=pilot, mech=pilot and agree >= AGREE_MIN)


rows = []
print(f"{'seed':>4} | {'normal: acc':>11} {'mirror':>7} {'near':>5} {'agreeARC0':>9} mech | "
      f"{'mirror init: acc':>16} {'mirror':>7} {'near':>5} {'agreeARC0':>9} mech | start residual")
for s in range(args.first, args.last + 1):
    wn, wm = load(s, "normal"), load(s, "mirror")
    if wn is None or wm is None:
        print(f"{s:>4} | missing ({'normal' if wn is None else ''} {'mirror' if wm is None else ''})")
        continue
    n, m = measure(wn), measure(wm)
    start = mirror_permutation(load(s, "mirror", 0))[1]
    rows.append((s, n, m))
    f = lambda r: f"{r['acc']:>11.3%} {r['res']:>7.3f} {r['near']:>5.0%} {r['agree']:>9.2%} {'YES' if r['mech'] else ' no'}"  # noqa: E731
    print(f"{s:>4} | {f(n)}  | {f(m):>40}  | {start:.1e}")

k = len(rows)
kn = sum(n["mech"] for _, n, _ in rows)
km = sum(m["mech"] for _, _, m in rows)
b = sum(m["mech"] and not n["mech"] for _, n, m in rows)  # mirror only
c = sum(n["mech"] and not m["mech"] for _, n, m in rows)  # normal only
p = sum(comb(b + c, i) for i in range(b, b + c + 1)) / 2 ** (b + c) if b + c else 1.0
print(f"\npairs complete: {k}")
print(f"mechanism: normal {kn}/{k}, mirror {km}/{k}; discordant pairs mirror-only {b}, normal-only {c}; "
      f"exact one-sided McNemar p = {p:.4g}")
sym_other = [s for s, _, m in rows if m["pilot"] and not m["mech"]]
left_sym = [s for s, _, m in rows if m["res"] >= MIRROR_MAX]
print(f"mirror arm: pass the pilot criterion but agree < 99 % with ARC 0 (another symmetric solution?): {sym_other or 'none'}")
print(f"mirror arm: left the symmetry (final residual >= 0.1): {left_sym or 'none'}")
if k == args.last - args.first + 1:
    alive = km >= MIN_MIRROR_ARM and p < ALPHA
    print(f"\nPRE-REGISTERED VERDICT: {'idea survives' if alive else 'idea dropped'} "
          f"(needs mirror >= {MIN_MIRROR_ARM}/{k} and p < {ALPHA}; got {km}/{k}, p = {p:.4g})")
else:
    print("\nnot all pairs trained yet: no verdict")
