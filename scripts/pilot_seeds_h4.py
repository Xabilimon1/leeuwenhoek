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
    < 0.5. Seed 0: 0.155; seeds 1-4: >= 1.13 at that point.
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
    print(f"{'seed':>4} {'exact acc':>10} {'near-tie*':>10} {'mirror':>7} {'mirror@2^21':>12} {'w_ih':>5}  mechanism  predicted")
    for r in rows:
        print(f"{r['seed']:>4} {r['acc']:>10.3%} {r['near']:>10.0%} {r['res']:>7.3f} {r['early']:>12.3f} {r['signs']:>5}  "
              f"{'YES' if r['mech'] else 'no':>9}  {'YES' if r['pred'] else 'no':>9}")
    print(f"* sampled ({args.n} inputs): share of errors with relative gap < {GAP}; base rate {np.mean(gap < GAP):.1%} of inputs")


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

# ---------------------------------------------------------------------------
# Post hoc (added after seeing the table above, NOT pre-registered).
from leeuwenhoek.structure import best_design_agreement, error_profile  # noqa: E402

print("\n--- post hoc, not pre-registered ---")
# 1) Same solution? The exact error per true ordering (6 numbers) does not change
#    under any relabelling or rescaling of neurons, so nets with the same
#    fingerprint fail on the same orderings in the same amounts.
nets = [(f"ARC {s}", arc(s, 2**30)) for s in range(5)] + [(str(r["seed"]), local(r["seed"], 2**30)) for r in rows]
from leeuwenhoek.exact import ORDERINGS  # noqa: E402

profiles = {name: error_profile(w)["error_by_order"] for name, w in nets}
prints = {name: np.array([p[o] for o in ORDERINGS]) for name, p in profiles.items()}
# The net w with w_ih negated computes f(-x): just as valid and just as likely
# (the initialization is sign-symmetric). -x reverses every ordering, so compare
# each fingerprint also with the other one read backwards.
flipped = {name: np.array([p[o[::-1]] for o in ORDERINGS]) for name, p in profiles.items()}


def dist(a, b):
    return min(np.abs(prints[a] - prints[b]).max(), np.abs(prints[a] - flipped[b]).max())


order = lambda s: (s.startswith("ARC"), int(s.split()[-1]))  # noqa: E731
groups = []
for name in prints:  # single linkage, 0.05 percentage points per ordering
    hit = [g for g in groups if any(dist(name, m) < 5e-4 for m in g)]
    merged = [name] + [m for g in hit for m in g]
    groups = [g for g in groups if g not in hit] + [merged]
print("nets with the same exact error fingerprint (error per ordering within 0.05 points, up to x -> -x):")
for g in sorted(groups, key=len, reverse=True):
    if len(g) > 1:
        print(f"  {', '.join(sorted(g, key=order))}   error by ordering, up to x -> -x (% of inputs): {np.round(100 * prints[g[0]], 2)}")
print(f"  singletons: {sorted((g[0] for g in groups if len(g) == 1), key=order)}")
print("nearest other net by fingerprint (largest difference over the 6 orderings, percentage points):")
for name in sorted(prints, key=order):
    other = min((m for m in prints if m != name), key=lambda m: dist(name, m))
    print(f"  {name:>6} -> {other:>6}: {100 * dist(name, other):.2f}")

# The fingerprint is necessary, not sufficient, for "same function": check the
# answers directly, taking the better of f(x) and f(-x) for the other net.
def agree(wa, wb):
    a = forward(wa, x).argmax(-1)
    return max(float(np.mean(a == forward(wb, x).argmax(-1))), float(np.mean(a == forward(wb, -x).argmax(-1))))


W = dict(nets)
print("functional agreement inside the near groups (same answer, best of x and -x) [sampled]:")
for a, b in [("13", "14"), ("13", "16"), ("13", "22"), ("5", "6"), ("12", "18"), ("10", "ARC 1"), ("7", "9"), ("7", "ARC 0")]:
    print(f"  {a:>5} vs {b:>5}: {agree(W[a], W[b]):.2%}")

# 2) Do the mechanism seeds fail where seed 0 fails? Reference: ARC seed 0 against
#    seed 0 retrained here (same initialization, same mechanism, not bit-identical).
w0 = arc(0, 2**30)
a0 = forward(w0, x).argmax(-1)
wrong0 = a0 != target
xs = x[:200_000]
d0 = best_design_agreement(w0, xs)
acc0 = exact_accuracy(w0)[0]
print(f"ARC seed 0: best seed-0 design agreement {d0['agreement']:.2%} -> agreement - accuracy {d0['agreement'] - acc0:+.2%}  [sampled]")
for name, w in [("seed 0 retrained here", local(0, 2**30))] + [(f"seed {r['seed']}", local(r["seed"], 2**30)) for r in rows if r["mech"]]:
    a = forward(w, x).argmax(-1)
    line = (f"{name}: same answer as ARC seed 0 on {np.mean(a == a0):.3%} of inputs; where ARC seed 0 is wrong, "
            f"same wrong answer {np.mean(a[wrong0] == a0[wrong0]):.1%}")
    if name != "seed 0 retrained here":
        d = best_design_agreement(w, xs)
        acc = exact_accuracy(w)[0]
        line += f"; design agreement - accuracy {d['agreement'] - acc:+.2%} (b/a={d['b_over_a']:.3f}, e={d['e']:.2f})"
    print(line + "  [sampled]")

# 3) The early predictor failed at 2^21: how does the mirror residual fall later?
print("mirror residual along training for the mechanism seeds:")
for r in [r for r in rows if r["mech"]]:
    traj = "  ".join(f"2^{k}:{mirror_permutation(local(r['seed'], 2**k))[1]:.3f}" for k in range(20, 31))
    print(f"  seed {r['seed']}: {traj}")

# 4) The near-chance nets (accuracy < 60 %): what do they answer?
for r in [r for r in rows if r["acc"] < 0.6]:
    w = local(r["seed"], 2**30)
    p = error_profile(w)
    ans = np.bincount(forward(w, x).argmax(-1), minlength=3) / len(x)
    print(f"  seed {r['seed']}: w_ih {np.round(w['w_ih'], 2)}  answers x1/x2/x3 {np.round(ans, 2)}  "
          f"dead final state {p['dead_share']:.0%} of errors")
