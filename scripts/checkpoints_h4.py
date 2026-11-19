"""Exact accuracy, input-weight signs and mirror residual of argmedian h=4 L=3 along ARC's training run.

For each AlgZoo seed (0-4) and each published checkpoint (n_seqs = 2^20 ... 2^29 and the final
2^30 model): exact accuracy from the weights (path enumeration, no sampling), the sign pattern of
w_ih, and the mirror residual (0 = exact sign mirror, 2 = none).

Question it answers: is the mechanism (signs, mirror) fixed early, and how does accuracy evolve?
Usage: PYTHONPATH=src .venv/bin/python scripts/checkpoints_h4.py
"""

import numpy as np

from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.structure import mirror_permutation

STEPS = [2**k for k in range(20, 30)] + [None]  # None = final model (2^30)

for seed in range(5):
    print(f"seed {seed}")
    for n in STEPS:
        _, w = load_argmedian(4, 3, seed=seed, n_seqs=n)
        acc = exact_accuracy(w)[0]
        signs = "".join("+" if v > 0 else "-" for v in w["w_ih"])
        _, res = mirror_permutation(w)
        tag = "2^30 (final)" if n is None else f"2^{int(np.log2(n))}"
        print(f"  {tag:>12}: accuracy {acc:.3%}  w_ih signs {signs}  mirror residual {res:.3f}")
