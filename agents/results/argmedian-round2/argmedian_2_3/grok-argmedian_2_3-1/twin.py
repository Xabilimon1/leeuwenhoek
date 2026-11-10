"""Hand-written rule matching the 2-unit arg-median RNN.

The network never answers 2. It answers 1 on two ascent cones
(region A: a non-positive first input, then a positive second that
the third exceeds by ratio k; region B: a positive first input that
the second overshoots, with the third in a band of that overshoot).
Everywhere else it answers 0.

Constants k, d, U are derived from the weights in EXPLANATION.md.
"""

import numpy as np

# k = -(w_hh[1] · w_ih) / w_ih[1]
#     comparison ratio: v fires on x_t > k * x_prev from a fresh positive
k = 0.8702462636

# d = -w_hh[1, 1]
#     v-self-inhibition: from a v-only state v = b*s, v fires on x_t > d*s
d = 1.9860576391

# U = b * (r_star * D - B) / (a - r_star * b)
#     readout crossing L0 = L1 in the v-only carry region
U = 20.02710476


def predict(x: np.ndarray) -> np.ndarray:
    """x: (n, 3) float -> (n,) ints in 0..2."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]

    # Region A: first input non-positive, then an ascent through position 1.
    region_a = (x0 <= 0.0) & (x1 > 0.0) & (x2 > k * x1)

    # Region B: first input positive, second overshoots it by ratio k,
    # third lands in (d, U) times the overshoot s = x1 - k x0.
    s = x1 - k * x0
    region_b = (x0 > 0.0) & (s > 0.0) & (x2 > d * s) & (x2 < U * s)

    return np.where(region_a | region_b, 1, 0).astype(np.int64)
