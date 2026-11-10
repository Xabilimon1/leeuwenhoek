"""Does the explanation of argmedian h=4 L=3 survive noise in the weights? (check suggested by ARC)

Each weight is multiplied by (1 + eps * z), z standard normal. For every noisy
network we compare
  - the mechanistic estimate rebuilt from the noisy weights (same mechanisms,
    paths and regions; the numbers inside come from the new weights), with
  - the exact accuracy of the noisy network (brute force, exact.py).
Baseline: keep the estimate of the original weights. If the explanation
generalizes, the rebuilt estimate follows the accuracy while the baseline drifts.
Coverage = share of the noisy network's error mass that lies inside the cones.

Usage: uv run python scripts/jacob_noise.py
"""

import numpy as np

from leeuwenhoek.cones import cone_fraction
from leeuwenhoek.estimate import error_of_cone
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.mechanisms import mechanism_cones
from leeuwenhoek.models import load_argmedian

LEVELS = [0.001, 0.003, 0.01, 0.03, 0.1]
DRAWS = 20


def estimate(w):
    cones = mechanism_cones(w)
    return 1 - sum(error_of_cone(n, s) for n, s in cones.values()), cones


def coverage(pieces, cones):
    err = [p for p in pieces if p["pred"] != p["order"][1]]
    inside = sum(min(p["prob"], sum(cone_fraction(list(p["normals"]) + list(n)) for n, _ in cones.values()))
                 for p in err)
    return inside / sum(p["prob"] for p in err)


_, w0 = load_argmedian(4, 3)
w0 = {k: v.astype(np.float64) for k, v in w0.items()}
exact0, _, pieces0 = exact_accuracy(w0)
est0, cones0 = estimate(w0)
print(f"original: exact {exact0:.4%}  estimate {est0:.4%}  error {100 * (est0 - exact0):+.3f} pts"
      f"  coverage {coverage(pieces0, cones0):.1%}\n")

rng = np.random.default_rng(2026)
print(f"{'eps':>6} | {'|accuracy change|':>17} | {'estimate error':>18} | {'baseline error':>14} | {'coverage':>8} | total")
print(f"{'':>6} | {'mean (max), pts':>17} | {'mean +- sd, pts':>18} | {'mean, pts':>14} | {'mean':>8} |")
for eps in LEVELS:
    rows = []
    for _ in range(DRAWS):
        w = {k: v * (1 + eps * rng.standard_normal(v.shape)) for k, v in w0.items()}
        exact, total, pieces = exact_accuracy(w)
        est, cones = estimate(w)
        rows.append((exact - exact0, est - exact, est0 - exact, coverage(pieces, cones), total))
    d, e, b, cov, tot = (np.array(c) for c in zip(*rows))
    print(f"{eps:>6} | {100 * np.abs(d).mean():>7.3f} ({100 * np.abs(d).max():>6.3f}) |"
          f" {100 * e.mean():>+8.3f} +- {100 * e.std():>5.3f} | {100 * np.abs(b).mean():>14.3f} |"
          f" {cov.mean():>8.1%} | {np.abs(tot - 1).max():.0e}")
