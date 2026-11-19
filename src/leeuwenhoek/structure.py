"""Seed-independent diagnostics of an argmedian L=3 ReLU RNN's structure.

Written to ask of seeds 1-4 the same questions we answered for seed 0:
where the errors are, whether the net has a sign mirror, what comparison each
neuron makes at each step, how much memory the state keeps, and whether the
seed-0 design (design.py) can describe the net at all.
"""

from itertools import permutations

import numpy as np

from leeuwenhoek.exact import ORDERINGS, exact_accuracy
from leeuwenhoek.forward import forward
from leeuwenhoek.linear import parse_path, path_matrices


def pattern_keys(states):
    """'1100-0011-1100' for each row of states (n, L, h): which neurons fire per step."""
    on = states > 0
    return np.array(["-".join("".join("1" if b else "0" for b in s) for s in row) for row in on])


def error_profile(w):
    """Exact error anatomy from the path enumeration (exact.py), no sampling.

    Returns accuracy, error, wrong_share (which answer the net gives when
    wrong), error_by_median (true median position when wrong),
    error_by_order (true ordering -> error mass), dead_share (errors from a
    dead final state: every logit 0, argmax -> x1) and the heaviest error
    pieces (ordering, answer, path, mass).
    """
    acc, total, pieces = exact_accuracy(w)
    assert abs(total - 1) < 1e-6, total
    wrong = [p for p in pieces if p["pred"] != p["order"][1]]
    err = sum(p["prob"] for p in wrong)
    wrong_share = np.zeros(3)
    by_median = np.zeros(3)
    by_order = {o: 0.0 for o in ORDERINGS}
    dead = 0.0
    for p in wrong:
        wrong_share[p["pred"]] += p["prob"]
        by_median[p["order"][1]] += p["prob"]
        by_order[p["order"]] += p["prob"]
        M, _ = path_matrices(w, parse_path(p["path"]))
        if np.abs(M).max() < 1e-12:
            dead += p["prob"]
    top = sorted(wrong, key=lambda p: -p["prob"])[:8]
    return {
        "accuracy": acc,
        "error": err,
        "wrong_share": wrong_share / err if err > 0 else wrong_share,
        "error_by_median": by_median,
        "error_by_order": by_order,
        "dead_share": dead / err if err > 0 else 0.0,
        "n_paths": len({p["path"] for p in pieces}),
        "top_pieces": [(p["order"], p["pred"], p["path"], p["prob"]) for p in top],
    }


def mirror_permutation(w):
    """Neuron permutation that best turns the net into its own sign mirror.

    A mirror P satisfies P w_ih = -w_ih, P W_hh P^T = W_hh, W_out P^T = W_out.
    Returns (perm, residual) with residual = the largest of the three relative
    errors. The identity scores 2 on w_ih, so a residual near 2 means no
    permutation does better than nothing; seed 0 scores 0.004.
    """
    w_ih, w_hh, w_out = (np.asarray(w[k], np.float64) for k in ("w_ih", "w_hh", "w_out"))
    best = None
    for perm in permutations(range(len(w_ih))):
        P = np.eye(len(w_ih))[list(perm)]
        r = max(
            np.linalg.norm(P @ w_ih + w_ih) / np.linalg.norm(w_ih),
            np.linalg.norm(P @ w_hh @ P.T - w_hh) / np.linalg.norm(w_hh),
            np.linalg.norm(w_out @ P.T - w_out) / np.linalg.norm(w_out),
        )
        if best is None or r < best[1]:
            best = (perm, float(r))
    return best


def symmetry_agreement(w, x):
    """Fraction of inputs where the answer for x and for -x is the same.

    The task itself is symmetric (the median position does not change under
    x -> -x), so a perfect net scores 1; seed 0 scores 99.6 %.
    """
    return float(np.mean(forward(w, x).argmax(-1) == forward(w, -x).argmax(-1)))


def comparison_from_patterns(keys, comparison):
    """How well the activation pattern determines a boolean comparison.

    Best lookup table pattern -> True/False, scored as accuracy. 1.0 means
    the pattern decides the comparison exactly; 0.5 means it says nothing.
    """
    keys = np.asarray(keys)
    comparison = np.asarray(comparison, bool)
    agree = 0
    for k in np.unique(keys):
        m = keys == k
        agree += max(comparison[m].sum(), (~comparison[m]).sum())
    return float(agree / len(keys))


def neuron_boundaries(w, prefix):
    """What each neuron compares at step t = len(prefix) + 1, given the prefix.

    Inside the prefix the pre-activation is linear, z = c . (x1..xt). Neuron i
    fires iff sign(c_t) (x_t - ref . (x1..x_{t-1})) > 0 with ref = -c[:t-1]/c_t:
    it checks whether x_t is "above" or "below" a reference built from the past.
    ref = (1,) at step 2 is a comparison with the previous value (seed 0:
    0.79 and 1.24); ref = (0,) only looks at the sign of x_t.
    Returns one dict per neuron: side, ref, coeffs (raw c), gain (|c_t|).
    If c_t == 0 the neuron ignores x_t: side = "none".
    """
    masks = list(prefix)
    t = len(masks)
    _, pre = path_matrices(w, masks + [np.ones(len(w["w_ih"]))], seq_len=t + 1)
    out = []
    for c in pre[t]:
        c = np.asarray(c, np.float64)[: t + 1]
        if abs(c[t]) < 1e-9:
            out.append({"side": "none", "ref": -c[:t], "coeffs": c, "gain": 0.0})
            continue
        out.append({
            "side": "above" if c[t] > 0 else "below",
            "ref": -c[:t] / c[t],
            "coeffs": c,
            "gain": float(abs(c[t])),
        })
    return out


def state_singular_values(w, prefix):
    """Singular values of the state after the prefix as a linear map of (x1..xt).

    Rank 2 after step 2 means the state still separates x1 from x2 (seed 0:
    8.7 for the comparison channel, ~1 for the memory channel); rank 1 means
    the net kept only one number; 0 means a dead state.
    """
    masks = list(prefix)
    t = len(masks)
    w_ih, w_hh = (np.asarray(w[k], np.float64) for k in ("w_ih", "w_hh"))
    h = np.zeros((len(w_ih), 3))
    for tt, d in enumerate(masks):
        z = w_hh @ h
        z[:, tt] += w_ih
        h = np.diag(d) @ z
    return np.linalg.svd(h[:, :t], compute_uv=False)


def best_design_agreement(w, x, ratios=None, es=None, a=8.67):
    """Best agreement between the net and any seed-0 design net (design.py).

    The design has 2 free numbers, b/a and e (a is the scale, B2/B3 do not
    change a single answer). We scan a grid and report the best agreement
    with the net's answers on x. Seed 0 reaches 98.8 % with its own numbers.
    """
    from leeuwenhoek.design import designed_weights

    ratios = np.linspace(0.02, 0.5, 25) if ratios is None else ratios
    es = np.linspace(0.1, 0.9, 17) if es is None else es
    x64 = x.astype(np.float64)
    ref = forward(w, x).argmax(-1)
    best = {"agreement": -1.0}
    for r in ratios:
        for e in es:
            d = designed_weights(a, r * a, e, dtype=np.float64)
            agr = float(np.mean(forward(d, x64).argmax(-1) == ref))
            if agr > best["agreement"]:
                best = {"agreement": agr, "b_over_a": float(r), "e": float(e)}
    return best
