"""FLOP count of the mechanistic estimate (mechanisms.py) versus sampling the model.

Convention: one FLOP per +, -, *, /, comparison, abs, sqrt, cos and arctan2;
a sort of n numbers costs n * ceil(log2 n) comparisons; copies, indexing and
sign flips of whole rows by masks are free except where counted. The counted
functions repeat the arithmetic of cones.py, linear.py, mechanisms.py and
estimate.py line by line, and return the same numbers bit for bit (tested),
so the count is of the computation that actually produced the estimate.
"""

from itertools import product
from math import ceil, log2

import numpy as np

from leeuwenhoek.cones import TOL
from leeuwenhoek.exact import ORDERINGS, ordering_normals
from leeuwenhoek.linear import parse_path
from leeuwenhoek.mechanisms import MECHANISMS, X1, mirror_path

CROSS, DOT3, NORM3 = 9, 5, 6  # 6 mul 3 sub; 3 mul 2 add; dot + sqrt


class Counter:
    def __init__(self):
        self.n = 0

    def add(self, k):
        self.n += int(k)


def _sort(n):
    return n * ceil(log2(n)) if n > 1 else 0


def forward_flops(h, L):
    """One model evaluation scored against the truth: forward pass, argmax, true median."""
    step = h + (h * h + h * (h - 1)) + h + h  # w_ih x, w_hh h, add, relu
    readout = L * h + L * (h - 1)
    return L * step + readout + (L - 1) + 3 + 1  # argmax; median of 3: 3 comparisons; compare


# --- cones.py, counted -----------------------------------------------------------


def _unit_rows(a, c):
    a = np.asarray(a, float).reshape(-1, 3)
    norms = np.linalg.norm(a, axis=1)
    c.add(NORM3 * len(a) + len(a))  # norms, threshold
    a = a[norms > 1e-12] / norms[norms > 1e-12, None]
    c.add(3 * len(a))
    return a


def _dedupe(vs, c):
    out = []
    for v in vs:
        for q in out:
            c.add(3 + 3 + 2 + 1)  # sub, abs, max of 3, threshold
            if np.abs(v - q).max() < 1e-7:
                break
        else:
            out.append(v)
    return np.array(out).reshape(-1, 3)


def _arc_fraction(normals, axis, c):
    e1 = np.cross(axis, [1.0, 0, 0])
    c.add(CROSS + NORM3 + 1)
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(axis, [0, 1.0, 0])
        c.add(CROSS)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    c.add(NORM3 + 3 + CROSS)
    k = len(normals)
    theta = np.arctan2(normals @ e2, normals @ e1)
    c.add(k * (2 * DOT3 + 1))
    cuts = np.sort(np.concatenate([theta + np.pi / 2, theta - np.pi / 2]) % (2 * np.pi))
    c.add(2 * k * 2 + _sort(2 * k))  # +-pi/2, mod
    cuts = np.append(cuts, cuts[0] + 2 * np.pi)
    c.add(1)
    total = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = (a + b) / 2
        c.add(2 + k * 3 + (k - 1))  # mid; sub, cos, compare per normal; all
        if (np.cos(mid - theta) > 0).all():
            total += b - a
            c.add(2)
    c.add(2)
    return total / (2 * np.pi)


def counted_cone_fraction(normals, c):
    """cones.cone_fraction, adding its FLOPs to counter c."""
    n = _dedupe(_unit_rows(normals, c), c)
    if len(n) == 0:
        return 1.0
    if len(n) == 1:
        return 0.5
    k = len(n)
    i, j = np.triu_indices(k, 1)
    r = np.cross(n[i], n[j])
    keep = np.linalg.norm(r, axis=1) > 1e-12
    c.add(len(i) * (CROSS + NORM3 + 1))
    if not keep.any():
        return 0.0
    r = r[keep] / np.linalg.norm(r[keep], axis=1, keepdims=True)
    c.add(keep.sum() * (NORM3 + 3))  # the norm is recomputed in cones.py
    cand = np.concatenate([r, -r])
    rays = _dedupe(cand[(cand @ n.T >= -TOL).all(1)], c)
    c.add(len(cand) * k * (DOT3 + 1) + len(cand) * (k - 1))  # dots, compares, all
    if len(rays) == 0:
        return 0.0
    if len(rays) == 2 and np.abs(rays[0] + rays[1]).max() < 1e-7:
        c.add(3 + 3 + 2 + 1)
        active = n[np.abs(n @ rays[0]) < 1e-7]
        c.add(k * (DOT3 + 2))
        return _arc_fraction(active, rays[0], c)
    if len(rays) == 2:
        c.add(3 + 3 + 2 + 1)
    if len(rays) < 3:
        return 0.0
    R = len(rays)
    centre = rays.mean(0)
    centre /= np.linalg.norm(centre)
    c.add(3 * (R - 1) + 3 + NORM3 + 3)
    e1 = rays[0] - (rays[0] @ centre) * centre
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    c.add(DOT3 + 3 + 3 + NORM3 + 3 + CROSS)
    rays = rays[np.argsort(np.arctan2(rays @ e2, rays @ e1))]
    c.add(R * (2 * DOT3 + 1) + _sort(R))
    r1, r2, r3 = rays[0], rays[1:-1], rays[2:]
    num = np.abs(np.cross(r2, r3) @ r1)
    den = 1 + r2 @ r1 + np.sum(r2 * r3, axis=1) + r3 @ r1
    T = R - 2
    c.add(T * (CROSS + DOT3 + 1) + T * (3 * DOT3 + 3) + T + (T - 1) + 3)  # num, den, atan2, sum, 2/(4pi)
    return float(2 * np.arctan2(num, den).sum() / (4 * np.pi))


# --- linear.py / exact.step_normals, counted -----------------------------------


def _path_matrices(w, masks, c, seq_len=None):
    w_ih, w_hh, w_out = (np.asarray(w[k], np.float64) for k in ("w_ih", "w_hh", "w_out"))
    L, hd = len(masks) if seq_len is None else seq_len, len(w_ih)
    h = np.zeros((hd, L))
    pre = []
    for t, d in enumerate(masks):
        z = w_hh @ h
        z[:, t] += w_ih
        pre.append(z)
        h = np.diag(d) @ z
        c.add(hd * L * (2 * hd - 1) + hd)  # the mask zeroes rows: free
    c.add(len(w_out) * L * (2 * hd - 1))
    return w_out @ h, pre


def _step_normals(pre_t, mask, c):
    c.add(len(pre_t) * (3 + 2 + 1))  # abs, max of 3, threshold per neuron
    return [z if on else -z for z, on in zip(pre_t, mask) if np.abs(z).max() > 1e-12]


# --- exact.exact_accuracy, counted (the brute-force baseline) -----------------


def exact_flops(w):
    """(exact accuracy, FLOPs) of exact.pieces, keeping only the pieces that predict
    the median: the leanest form of the brute force, so the comparison favours it."""
    w = {k: np.asarray(v, np.float64) for k, v in w.items()}
    c = Counter()
    masks = [np.array(m, float) for m in product((0, 1), repeat=len(w["w_ih"]))]

    def extend(prefix, normals):
        t = len(prefix)
        if t == 3:
            yield prefix, normals
            return
        for m in masks:
            _, pre = _path_matrices(w, prefix + [m], c, seq_len=3)
            new = normals + _step_normals(pre[t], m, c)
            if counted_cone_fraction(new, c) > 0:
                yield from extend(prefix + [m], new)

    acc = 0
    for path, normals in extend([], []):
        M, _ = _path_matrices(w, path, c)
        for o in ORDERINGS:
            base = normals + ordering_normals(o)
            if counted_cone_fraction(base, c) == 0:
                continue
            k = o[1]
            diffs = [M[k] - M[j] for j in range(3) if j != k]
            c.add(2 * 3 + 2 * (3 + 2 + 1))  # differences, all-zero test
            if all(np.abs(d).max() < 1e-12 for d in diffs):
                if k != 0:
                    continue
                cone = base
            else:
                cone = base + diffs
            p = counted_cone_fraction(cone, c)
            if p > 0:
                acc += p
                c.add(1)
    return acc, c.n


# --- mechanisms.mechanism_cones + estimate.error_of_cone, counted ---------------


def estimate_flops(w):
    """(estimate, total FLOPs, FLOPs by stage) for the mechanistic estimate from weights w."""
    c_build, c_dead, c_prob = Counter(), Counter(), Counter()
    cones = {}
    for name, (path, region, says, rivals) in MECHANISMS.items():
        for side, sign, p in [("x1>0", 1, path), ("x1<0", -1, mirror_path(path))]:
            M, _ = _path_matrices(w, parse_path(p), c_build)
            normals = [sign * np.asarray(n, float) for n in region]
            normals += [M[says] - M[r] for r in rivals]
            c_build.add(3 * len(rivals))
            cones[f"{name} ({side})"] = (normals, says)
    for m2 in product("01", repeat=4):
        for side, p in [("x1>0", f"1100-{''.join(m2)}-0000"), ("x1<0", mirror_path(f"1100-{''.join(m2)}-0000"))]:
            masks = parse_path(p)
            _, pre = _path_matrices(w, masks, c_dead)
            normals = [n for t in range(len(masks)) for n in _step_normals(pre[t], masks[t], c_dead)]
            if counted_cone_fraction(normals, c_dead) > 0:
                c_dead.add(1)
                cones[f"dead, path {p} ({side})"] = (normals, X1)
    errs = []
    for normals, says in cones.values():
        parts = [counted_cone_fraction(list(normals) + ordering_normals(o), c_prob)
                 for o in ORDERINGS if o[1] != says]
        errs.append(sum(parts))
        c_prob.add(len(parts) - 1)
    est = 1 - sum(errs)
    c_prob.add(len(errs))
    stages = {"build mechanism cones": c_build.n, "find dead paths": c_dead.n,
              "probabilities of the cones": c_prob.n}
    return est, sum(stages.values()), stages
