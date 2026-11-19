"""Do the other seeds of argmedian h=4 L=3 use the seed-0 mechanism?

AlgZoo trains 5 seeds and ships the best (seed 0). For each seed this prints
the exact error anatomy, the sign mirror (or its absence), what every neuron
compares at each step, how much memory the state keeps, the main activation
paths with their logit maps, and the best agreement any seed-0 design net
(design.py, 2 free numbers) can reach. Exact numbers come from the path
enumeration; the few sampled ones are marked (n inputs, seed 1).

Usage: uv run python scripts/seeds_h4.py [--seeds 0 1 2 3 4] [--n 1000000] [--no-design]
"""

import argparse
from collections import Counter

import numpy as np

from leeuwenhoek.design import designed_weights, read_off
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.forward import forward
from leeuwenhoek.linear import parse_path, path_matrices
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.structure import (
    best_design_agreement,
    comparison_from_patterns,
    error_profile,
    mirror_permutation,
    neuron_boundaries,
    pattern_keys,
    state_singular_values,
    symmetry_agreement,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--n", type=int, default=1_000_000)
parser.add_argument("--no-design", action="store_true")
args = parser.parse_args()
np.set_printoptions(precision=2, suppress=True, linewidth=130)

x = gaussian_inputs(args.n, 3, seed=1)
target = argmedian_target(x)
gap = np.diff(np.sort(x, -1), axis=-1).min(-1) / np.abs(x).max(-1)  # smallest relative gap
z_mean = x.sum(-1) / (np.sqrt(3) * np.linalg.norm(x, axis=-1))  # sphere.py's z: +-1 = all three values equal
comparisons = {
    "x1>0": x[:, 0] > 0, "x2>x1": x[:, 1] > x[:, 0], "x3>x2": x[:, 2] > x[:, 1],
    "x3>x1": x[:, 2] > x[:, 0], "x2>0": x[:, 1] > 0, "x3>0": x[:, 2] > 0,
}
order_name = lambda o: "<".join(f"x{i + 1}" for i in o)  # noqa: E731
_, w0 = load_argmedian(4, 3, seed=0)
ro = read_off(w0)
summary = []

for seed in args.seeds:
    _, w = load_argmedian(4, 3, seed=seed)
    print(f"\n{'=' * 30} seed {seed} {'=' * 30}")
    print("w_ih ", w["w_ih"])
    print("W_hh\n", w["w_hh"])
    print("W_out (rows = logits x1 x2 x3)\n", w["w_out"])

    # --- exact error anatomy
    p = error_profile(w)
    print(f"\nexact accuracy {p['accuracy']:.5%}   error {p['error']:.4%}   paths {p['n_paths']}")
    print(f"  wrong answer is x1/x2/x3: {np.round(p['wrong_share'], 3)}   dead final state: {p['dead_share']:.1%} of errors")
    print(f"  true median was x1/x2/x3 when wrong (share of errors): {np.round(p['error_by_median'] / p['error'], 3)}")
    print("  error by true ordering (each ordering holds 16.67 % of inputs):")
    for o, e in sorted(p["error_by_order"].items(), key=lambda kv: -kv[1]):
        print(f"    {order_name(o)} (median x{o[1] + 1}): {e:.3%} of inputs = {6 * e:.1%} of that ordering")
    print("  heaviest error pieces (ordering, answer, path, mass):")
    for o, k, path, m in p["top_pieces"][:6]:
        print(f"    {order_name(o)} -> x{k + 1}  {path}  {m:.3%}")

    # --- sampled: near ties, symmetry
    logits, states = forward(w, x, return_states=True)
    pred = logits.argmax(-1)
    wrong = pred != target
    print(f"\nsampled (n = {args.n}): accuracy {1 - wrong.mean():.3%}")
    for thr in (0.05, 0.2):
        print(f"  errors with relative gap < {thr}: {np.mean(gap[wrong] < thr):.1%}  (base rate {np.mean(gap < thr):.1%})")
    for thr in (0.8, 0.9):  # polar caps of the sphere map: the three values alike and of one sign
        print(f"  errors with |z| > {thr} (z = mean part): {np.mean(np.abs(z_mean[wrong]) > thr):.1%}  (base rate {np.mean(np.abs(z_mean) > thr):.1%})"
              f"   of which z > 0: {np.mean(z_mean[wrong & (np.abs(z_mean) > thr)] > 0):.0%}")
    sym = symmetry_agreement(w, x)
    print(f"  answer(x) == answer(-x): {sym:.2%}   accuracy x1>0: {1 - wrong[x[:, 0] > 0].mean():.2%}  x1<0: {1 - wrong[x[:, 0] < 0].mean():.2%}")
    perm, res = mirror_permutation(w)
    print(f"  best mirror permutation {tuple(i + 1 for i in perm)}: residual {res:.3f}  (0 = exact mirror, 2 = none)")

    # --- activation patterns, dead / always-on neurons, comparisons
    on = states > 0
    rate = on.mean(0)
    print("\nfiring rate per neuron (rows = steps):\n", rate)
    for t in range(3):
        for i in range(4):
            if rate[t, i] < 0.01:
                print(f"  step {t + 1}: neuron {i + 1} is dead (fires {rate[t, i]:.2%})")
            if rate[t, i] > 0.99:
                print(f"  step {t + 1}: neuron {i + 1} is always on (fires {rate[t, i]:.2%}) -> linear, no gate")
    for t in range(3):
        c = Counter(pattern_keys(states[:, t : t + 1]))
        pats = "  ".join(f"{k}:{v / args.n:.1%}" for k, v in c.most_common(8) if v / args.n >= 0.005)
        print(f"  step {t + 1} patterns: {pats}")
        keys_t = pattern_keys(states[:, : t + 1])  # the whole prefix, as the net itself has it
        det = "  ".join(f"{name}:{comparison_from_patterns(keys_t, cmp):.3f}" for name, cmp in comparisons.items())
        print(f"         prefix pattern determines  {det}")

    # --- what each neuron compares (step 2 for both signs of x1; step 3 for the main prefixes)
    print("\nneuron boundaries (fires iff x_t is above/below ref . past):")
    prefixes2 = [k for k, _ in Counter(pattern_keys(states[:, :1])).most_common(2)]
    for pre in prefixes2:
        sign = "x1>0" if (x[pattern_keys(states[:, :1]) == pre][:, 0] > 0).mean() > 0.5 else "x1<0"
        print(f"  step 2 after {pre} ({sign}):")
        for i, b in enumerate(neuron_boundaries(w, parse_path(pre))):
            if b["side"] == "none":
                print(f"    n{i + 1}: ignores x2 (pre = {b['coeffs'][0]:.2f} x1)")
            else:
                print(f"    n{i + 1}: on iff x2 {'>' if b['side'] == 'above' else '<'} {b['ref'][0]:.2f} x1   (gain {b['gain']:.1f})")
    prefixes3 = [k for k, v in Counter(pattern_keys(states[:, :2])).most_common(4) if v / args.n > 0.05]
    for pre in prefixes3:
        sv = state_singular_values(w, parse_path(pre))
        print(f"  step 3 after {pre}: state singular values {np.round(sv, 2)} (comparison, memory)")
        for i, b in enumerate(neuron_boundaries(w, parse_path(pre))):
            if b["side"] == "none":
                print(f"    n{i + 1}: ignores x3")
            else:
                r = b["ref"]
                print(f"    n{i + 1}: on iff x3 {'>' if b['side'] == 'above' else '<'} {r[0]:.2f} x1 + {r[1]:.2f} x2   (gain {b['gain']:.1f})")

    # --- main final paths and their logit maps
    keys = pattern_keys(states)
    cnt = Counter(keys)
    print("\nmain paths (share, error inside, true orderings; logit differences as rows on (x1, x2, x3)):")
    for k, c in cnt.most_common(8):
        m = keys == k
        M, _ = path_matrices(w, parse_path(k))
        orders = Counter(tuple(np.argsort(r)) for r in x[m])
        od = " ".join(f"{order_name(o)}:{v / m.sum():.2f}" for o, v in orders.most_common(3))
        print(f"  {k}: {c / args.n:.1%}  err {wrong[m].mean():.1%}  {od}")
        print(f"      L2-L1 {np.round(M[1] - M[0], 1)}   L3-L1 {np.round(M[2] - M[0], 1)}   L3-L2 {np.round(M[2] - M[1], 1)}")

    # --- can the seed-0 design describe it?
    design = None
    if not args.no_design:
        xs = x[:200_000]
        own = best_design_agreement(w, xs, ratios=(ro["b"] / ro["a"],), es=(ro["e"],))["agreement"]
        design = best_design_agreement(w, xs)
        d_acc = exact_accuracy(designed_weights(8.67, 8.67 * design["b_over_a"], design["e"], dtype=np.float64))[0]
        design["own"], design["accuracy"] = own, d_acc
        print(f"\ndesign (2 free numbers): agreement with seed-0 numbers (b/a={ro['b'] / ro['a']:.3f}, e={ro['e']:.3f}) {own:.2%};"
              f" best on the grid {design['agreement']:.2%} at b/a={design['b_over_a']:.3f}, e={design['e']:.2f}  [sampled, 200k]")
        print(f"  agreement - net accuracy = {design['agreement'] - p['accuracy']:+.2%} (0 = the design only agrees where the net is right)."
              f"  That best design has exact accuracy {d_acc:.2%} on its own.")
    summary.append((seed, p, sym, res, design))

print(f"\n{'=' * 30} summary {'=' * 30}")
print(f"{'seed':>4} {'exact acc':>10} {'wrong->x1':>9} {'dead':>6} {'sym(x,-x)':>9} {'mirror':>7} {'design best':>11} {'- acc':>7}")
for seed, p, sym, res, design in summary:
    d = f"{design['agreement']:.1%}" if design else "-"
    dd = f"{design['agreement'] - p['accuracy']:+.2%}" if design else "-"
    print(f"{seed:>4} {p['accuracy']:>10.3%} {p['wrong_share'][0]:>9.1%} {p['dead_share']:>6.1%} {sym:>9.1%} {res:>7.3f} {d:>11} {dd:>7}")
