"""Where do the designed nets fail, and does that region shrink in one or two directions?

For designed nets (design.py) with shrinking memory b, measure on the errors:
  - polar angle: angle between x and the diagonal (1,1,1) (0 = all three equal)
  - gaps: the two smallest of |x_i - x_j| relative to |x|
If errors sit in a cap around the diagonal whose radius scales like b/a, the
error probability scales like (b/a)^2.

Usage: uv run python scripts/law_geometry.py
"""

import numpy as np

from leeuwenhoek.design import designed_weights, read_off
from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian

_, w = load_argmedian(4, 3)
p = read_off(w)
x = gaussian_inputs(4_000_000, 3, seed=14).astype(np.float64)
norm = np.linalg.norm(x, axis=1)
angle = np.degrees(np.arccos(np.clip(np.abs(x.sum(1)) / (np.sqrt(3) * norm), -1, 1)))
srt = np.sort(x, axis=1)
gap_small = np.minimum(srt[:, 1] - srt[:, 0], srt[:, 2] - srt[:, 1]) / norm
gap_big = np.maximum(srt[:, 1] - srt[:, 0], srt[:, 2] - srt[:, 1]) / norm
target = argmedian_target(x)

print(f"{'b/a':>7} {'error':>8} {'angle to diagonal (deg)':>26} {'small gap':>18} {'big gap':>18}")
print(f"{'':>7} {'':>8} {'median / 95th pct':>26} {'median / 95th':>18} {'median / 95th':>18}")
for b in [0.4, 0.2, 0.1, 0.05]:
    wd = designed_weights(p["a"], b, p["e"], p["B2"], p["B3"], dtype=np.float64)
    wrong = forward(wd, x).argmax(-1) != target
    r = b / p["a"]
    q = lambda v: f"{np.median(v[wrong]):.4f} / {np.percentile(v[wrong], 95):.4f}"
    print(f"{r:7.4f} {wrong.mean():8.4%} {q(angle):>26} {q(gap_small):>18} {q(gap_big):>18}")
print(f"\nall inputs: angle median {np.median(angle):.1f} deg; small gap median {np.median(gap_small):.3f}")

# Prediction from the algebra: inside the broken-pair zone the readout swaps
# its channels, which shifts the decision by ~ (b / (e a))^2 |x|. So the error
# should be K' (b / (e a))^2 with K' independent of e. Check with exact errors.
from leeuwenhoek.exact import exact_accuracy  # noqa: E402

print("\nexact error of designed nets at b = 0.05:")
print(f"{'e':>6} {'error':>10} {'error / (b/a)^2':>16} {'error / (b/(e a))^2':>20}")
for e in [0.25, 0.3, 0.369, 0.45, 0.55, 0.7]:
    b = 0.05
    err = 1 - exact_accuracy(designed_weights(p["a"], b, e, p["B2"], p["B3"], dtype=np.float64))[0]
    print(f"{e:6.3f} {err:10.5%} {err / (b / p['a']) ** 2:16.3f} {err / (b / (e * p['a'])) ** 2:20.4f}")
