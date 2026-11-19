"""Train argmedian h=4 L=3 with AlgZoo's own recipe on this machine and compare it with ARC's seed.

Asked by Prof. Javier Sánchez-Soriano (UFV): does the seed-0 mechanism appear
when networks are trained from scratch? Before training many seeds we check
(1) how long one seed takes here and (2) whether training here reproduces
ARC's network for the same seed.

Result on 2026-11-19 (MacBook Air M3, CPU, 1 thread), seed 0:
  - 316 s for the full 2^30 sequences;
  - the step-0 checkpoint is bit-identical to ARC's (same initialization);
  - the final net is not bit-identical but has the same mechanism: w_ih
    [3.96, 4.70, -4.71, -3.96] vs ARC [3.97, 4.70, -4.72, -3.97], mirror
    residual 0.003, exact accuracy 98.600 % vs 98.455 %.

Checkpoints go to data/trained/argmedian_4_3_seed<s>/ (gitignored).

Usage: uv run python scripts/train_seed_h4.py --seed 0 [--compare-only]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch as th
from alg_zoo.architectures import DistRNN

from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.models import load_argmedian, numpy_weights
from leeuwenhoek.structure import mirror_permutation

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--compare-only", action="store_true")
args = parser.parse_args()

out = Path(__file__).resolve().parents[1] / "data" / "trained" / f"argmedian_4_3_seed{args.seed}"
if not args.compare_only:
    out.mkdir(parents=True, exist_ok=True)
    os.environ["RESULTS_DIR"] = str(out)
    th.set_num_threads(1)
    from alg_zoo.training import train

    t = time.time()
    train("argmedian", hidden_size=4, seq_len=3, seed=args.seed, log_every=2**16)
    print(f"training time: {time.time() - t:.0f} s")


def load(path):
    m = DistRNN(hidden_size=4, seq_len=3)
    m.load_state_dict(th.load(path, weights_only=True))
    return m


final = load(out / f"argmedian_4_3_{2**30}_{args.seed}.pth")
w = numpy_weights(final)
_, w_arc = load_argmedian(4, 3, seed=args.seed)

from alg_zoo.zoo import zoo_argmedian  # noqa: E402

init_here = load(out / f"argmedian_4_3_0_{args.seed}.pth").state_dict()
init_arc = zoo_argmedian(4, 3, seed=args.seed, n_seqs=0).state_dict()
same_init = all(th.equal(init_here[k], init_arc[k]) for k in init_arc)

np.set_printoptions(precision=2, suppress=True)
print(f"seed {args.seed}: initialization identical to ARC's step-0 checkpoint: {same_init}")
print(f"  final weights identical to ARC's: {all(np.array_equal(w[k], w_arc[k]) for k in w)}")
for name, ww in [("here", w), ("ARC ", w_arc)]:
    perm, res = mirror_permutation(ww)
    print(f"  {name}: exact accuracy {exact_accuracy(ww)[0]:.3%}  w_ih {ww['w_ih']}  mirror residual {res:.3f}")
diff = max(float(np.abs(w[k] - w_arc[k]).max()) for k in w)
print(f"  largest weight difference here vs ARC: {diff:.3f}")
