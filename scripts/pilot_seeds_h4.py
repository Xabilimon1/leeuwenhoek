"""Pilot: how often does training from scratch find the seed-0 mechanism of argmedian h=4 L=3?

Asked by Prof. Javier Sánchez-Soriano (UFV). Seeds 5-24 are trained here with
AlgZoo's own recipe (scripts/train_seed_h4.py --train-only, 1 thread each);
seeds 0-4 are ARC's and are shown apart, because seed 0 is the one we
explained and ARC kept it as the best of 5 (a selected seed, not a random one).

Criteria fixed on 2026-11-19 before looking at any of seeds 5-24
(notes/argmedian_4_3_seeds.md, "Piloto de 20 semillas"):
  - mechanism present  <=>  final mirror residual < 0.1  AND  >= 80 % of the
    errors at near ties (relative gap < 0.05). Seed 0: 0.004 and 96 %;
    seeds 1-4: 1.03-1.41 and 13-47 %.
  - early predictor: mirror residual at 2^21 sequences (0.2 % of training)
    < 0.5. Seed 0: 0.155; seeds 1-4: >= 1.16 at that point.
The rate is reported with an exact (Clopper-Pearson) 95 % interval.

Usage: PYTHONPATH=src python scripts/pilot_seeds_h4.py [--seeds 5 ... 24] [--n 1000000]
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

MIRROR_MAX, NEAR_TIE_MIN, GAP, EARLY_N, EARLY_MAX = 0.1, 0.8, 0.05, 2**21, 0.5
ROOT = Path(__file__).resolve().parents[1] / "data" / "trained"

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5, 25)))
parser.add_argument("--n", type=int, default=1_000_000)
args = parser.parse_args()

x = gaussian_inputs(args.n, 3, seed=1)
target = argmedian_target(x)
gap = np.diff(np.sort(x, -1), axis=-1).min(-1) / np.abs(x).max(-1)  # same as seeds_h4.py


def local(seed, n_seqs):
    m = DistRNN(hidden_size=4, seq_len=3)
    m.load_state_dict(th.load(ROOT / f"argmedian_4_3_seed{seed}" / f"argmedian_4_3_{n_seqs}_{seed}.pth", weights_only=True))
    return numpy_weights(m)


def arc(seed, n_seqs):
    return load_argmedian(4, 3, seed=seed, n_seqs=None if n_seqs == 2**30 else n_seqs)[1]


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval by bisection on the binomial tail (no scipy here)."""
    def tail_ge(p):  # P(X >= k)
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def solve(f, target):  # f increasing in p
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
        return (lo + hi) / 2

    low = 0.0 if k == 0 else solve(tail_ge, alpha / 2)
    high = 1.0 if k == n else solve(lambda p: -(1 - tail_ge(p) + comb(n, k) * p**k * (1 - p) ** (n - k)), -alpha / 2)
    return low, high


def measure(seed, get):
    w = get(seed, 2**30)
    acc = exact_accuracy(w)[0]
    wrong = forward(w, x).argmax(-1) != target
    near = float(np.mean(gap[wrong] < GAP)) if wrong.any() else float("nan")
    res = mirror_permutation(w)[1]
    early = mirror_permutation(get(seed, EARLY_N))[1]
    signs = "".join("+" if v > 0 else "-" for v in w["w_ih"])
    mech = res < MIRROR_MAX and near >= NEAR_TIE_MIN
    return dict(seed=seed, acc=acc, near=near, res=res, early=early, signs=signs, mech=mech, pred=early < EARLY_MAX)


def table(rows, title):
    print(f"\n{title}")
    print(f"{'seed':>4} {'exact acc':>10} {'near-tie':>9} {'mirror':>7} {'mirror@2^21':>12} {'w_ih':>5}  mechanism  predicted")
    for r in rows:
        print(f"{r['seed']:>4} {r['acc']:>10.3%} {r['near']:>9.0%} {r['res']:>7.3f} {r['early']:>12.3f} {r['signs']:>5}  "
              f"{'YES' if r['mech'] else 'no':>9}  {'YES' if r['pred'] else 'no':>9}")


table([measure(s, arc) for s in range(5)], "ARC seeds 0-4 (reference; seed 0 = ARC's best of 5)")
rows = [measure(s, local) for s in args.seeds]
table(rows, f"trained here: seeds {args.seeds[0]}-{args.seeds[-1]}")

n, k = len(rows), sum(r["mech"] for r in rows)
lo, hi = clopper_pearson(k, n)
print(f"\nmechanism found in {k} of {n} seeds = {k / n:.0%}  (95 % interval {lo:.0%}-{hi:.0%})")
tp = sum(r["mech"] and r["pred"] for r in rows)
fp = sum(r["pred"] and not r["mech"] for r in rows)
fn = sum(r["mech"] and not r["pred"] for r in rows)
print(f"early predictor (mirror@2^21 < {EARLY_MAX}): {tp} hits, {fp} false alarms, {fn} misses")
borderline = [r["seed"] for r in rows if (r["res"] < MIRROR_MAX) != (r["near"] >= NEAR_TIE_MIN)]
print(f"seeds passing only one of the two tests: {borderline or 'none'}")
accs = np.array([r["acc"] for r in rows])
print(f"exact accuracy: median {np.median(accs):.2%}, range {accs.min():.2%}-{accs.max():.2%}")
