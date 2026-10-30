"""Exact accuracy of a bias-free ReLU DistRNN with L=3, from the weights alone.

The input space splits into pieces (activation path) x (true ordering) x
(predicted position). Each piece is a convex polyhedral cone, so its Gaussian
probability is exact (cones.py). Summing the pieces where the prediction is the
median gives the exact accuracy. Paths are enumerated step by step and a prefix
is dropped as soon as its cone is empty.

This is brute force, not an explanation: it is the ground truth against which
explanations and estimates are scored.
"""

from itertools import permutations, product

import numpy as np

from leeuwenhoek.cones import cone_fraction
from leeuwenhoek.linear import path_matrices

ORDERINGS = list(permutations(range(3)))  # (smallest, median, largest)


def ordering_normals(o):
    a, b, c = o
    n1, n2 = np.zeros(3), np.zeros(3)
    n1[[a, b]] = (-1, 1)  # x_b > x_a
    n2[[b, c]] = (-1, 1)  # x_c > x_b
    return [n1, n2]


def step_normals(pre_t, mask):
    """Half-spaces saying neuron i is on (pre > 0) or off (pre < 0) at this step."""
    return [z if on else -z for z, on in zip(pre_t, mask) if np.abs(z).max() > 1e-12]


def pieces(w):
    """Yield dicts with path, ordering, predicted position and exact probability."""
    w = {k: np.asarray(v, np.float64) for k, v in w.items()}
    h = len(w["w_ih"])
    masks = [np.array(m, float) for m in product((0, 1), repeat=h)]

    def extend(prefix, normals):
        t = len(prefix)
        if t == 3:
            yield prefix, normals
            return
        for m in masks:
            _, pre = path_matrices(w, prefix + [m], seq_len=3)
            new = normals + step_normals(pre[t], m)
            if cone_fraction(new) > 0:
                yield from extend(prefix + [m], new)

    for path, normals in extend([], []):
        M, _ = path_matrices(w, path)
        name = "-".join("".join(str(int(b)) for b in m) for m in path)
        for o in ORDERINGS:
            base = normals + ordering_normals(o)
            if cone_fraction(base) == 0:
                continue
            for k in range(3):
                diffs = [M[k] - M[j] for j in range(3) if j != k]
                if all(np.abs(d).max() < 1e-12 for d in diffs):
                    # all logits equal (dead state): argmax picks the first index
                    if k != 0:
                        continue
                    p = cone_fraction(base)
                else:
                    # ties go to the lower index; measure zero, ignore
                    p = cone_fraction(base + diffs)
                if p > 0:
                    yield {"path": name, "order": o, "pred": k, "prob": p}


def exact_accuracy(w):
    ps = list(pieces(w))
    acc = sum(p["prob"] for p in ps if p["pred"] == p["order"][1])
    total = sum(p["prob"] for p in ps)
    return acc, total, ps
