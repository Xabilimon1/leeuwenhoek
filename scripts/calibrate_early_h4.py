"""Calibrate an early predictor of the seed-0 mechanism on the 25 nets we already know.

The pilot's pre-registered predictor (mirror residual at 2^21 < 0.5) missed both
mechanism seeds. Choosing a new point by looking at the same seeds and then
claiming it works would be circular, so this script only CALIBRATES on the 25
known nets (ARC seeds 0-4 + local seeds 5-24); the chosen point is then
pre-registered and tested on seeds 25-124, which nobody has looked at.

For every checkpoint it prints the largest mirror residual among the mechanism
nets and the smallest among the others; the gap between them is the room a
threshold has.

Usage: PYTHONPATH=src python scripts/calibrate_early_h4.py
"""

from pathlib import Path

import numpy as np
import torch as th
from alg_zoo.architectures import DistRNN

from leeuwenhoek.models import load_argmedian, numpy_weights
from leeuwenhoek.structure import mirror_permutation

ROOT = Path(__file__).resolve().parents[1] / "data" / "trained"
MECH = {("ARC", 0), ("local", 7), ("local", 9)}  # from scripts/pilot_seeds_h4.py (commit 7ea9cf2)
NETS = [("ARC", s) for s in range(5)] + [("local", s) for s in range(5, 25)]


def weights(kind, seed, n):
    if kind == "ARC":
        return load_argmedian(4, 3, seed=seed, n_seqs=None if n == 2**30 else n)[1]
    m = DistRNN(hidden_size=4, seq_len=3)
    m.load_state_dict(th.load(ROOT / f"argmedian_4_3_seed{seed}" / f"argmedian_4_3_{n}_{seed}.pth", weights_only=True))
    return numpy_weights(m)


res = {net: [mirror_permutation(weights(*net, 2**k))[1] for k in range(20, 31)] for net in NETS}
print(f"{'checkpoint':>10} {'max mech':>9} {'min other':>10} {'gap':>7}  mechanism nets (ARC 0, 7, 9)")
for i, k in enumerate(range(20, 31)):
    mech = [res[n][i] for n in NETS if n in MECH]
    other = [res[n][i] for n in NETS if n not in MECH]
    gap = min(other) - max(mech)
    print(f"{'2^' + str(k):>10} {max(mech):>9.3f} {min(other):>10.3f} {gap:>+7.3f}  " + "  ".join(f"{v:.3f}" for v in mech))
print("\nsmallest residual among the other 22 nets at each checkpoint, and which net:")
for i, k in enumerate(range(20, 25)):
    n = min((n for n in NETS if n not in MECH), key=lambda n: res[n][i])
    print(f"  2^{k}: {res[n][i]:.3f} ({n[0]} {n[1]})")
