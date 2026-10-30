"""Close the gap: predict each kind of error of the trained h=4 net to first order.

Every error of this net sits next to a tie x_i = x_j. To first order, the
probability of a thin window next to a tie is
    P = p_d(0) * E[ width ],   p_d(0) = 1 / (2 sqrt(pi))   (d = x_i - x_j ~ N(0, 2))
with the expectation over the pair mean m ~ N(0, 1/2) and the third value
z ~ N(0, 1). The widths are read off the (mirror-symmetrized) weights:
  turn         x1 ~ x3: the model's x1/x3 boundary is shifted   (turn_slivers.py)
  step 2       x1 ~ x2: the model's x1/x2 boundary is shifted   (tie_band_step2.py)
  band + dead  x2 ~ x3 with x1 extreme: the window where x1 wins, either in the
               mixed path (logit edges, tie_band.py) or because every step-3
               neuron is off (dead path pre-activations)
We do x1 > 0 and double it (the symmetrized net is exactly mirror symmetric);
the trained net's own mirror asymmetry is added as measured.
Expectations are 2-D quadratures on a grid: nothing here runs the network.

Usage: uv run python scripts/gap_formula.py
"""

import numpy as np

from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.linear import parse_path, path_matrices
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import symmetrize

_, w = load_argmedian(4, 3)
s = {k: v.astype(np.float64) for k, v in symmetrize(w).items()}
PD0 = 1 / (2 * np.sqrt(np.pi))

# quadrature grid for m ~ N(0, 1/2) and z ~ N(0, 1)
g = np.linspace(-7, 7, 1401)
m = g[:, None] / np.sqrt(2)
z = g[None, :]
weight = np.exp(-g**2 / 2)[:, None] * np.exp(-g**2 / 2)[None, :]
weight /= weight.sum()


def expect(f):
    return float((f * weight).sum())


def boundary(path, a, b):
    """Coefficients (k1, k2, k3) of L_a - L_b = k . x on a path."""
    M, _ = path_matrices(s, parse_path(path))
    return M[a] - M[b]


# ---- turn: tie x1 ~ x3 (d = x3 - x1); at the tie x1 = x3 = m, third value z = x2
def shift_x3(path):
    # model boundary L_x1 = L_x3 solved for x3 at x1 = m: d* = x3* - m
    k = boundary(path, 0, 2)
    return lambda m, z: -(k[0] * m + k[1] * z) / k[2] - m


turn_min, turn_max = shift_x3("1100-0011-1100"), shift_x3("1100-1100-0011")
P_turn = PD0 * expect(np.where(m > 0, np.where(z < m, np.abs(turn_min(m, z)), np.abs(turn_max(m, z))), 0))

# ---- step 2: tie x1 ~ x2 (d = x2 - x1); at the tie x1 = x2 = m, third value z = x3
def shift_x2(path):
    k = boundary(path, 0, 1)  # L_x1 - L_x2 = 0 solved for x2 at x1 = m
    return lambda m, z: -(k[0] * m + k[2] * z) / k[1] - m


above, below = shift_x2("1100-0101-1100"), shift_x2("1100-0101-0011")
P_step2 = PD0 * expect(np.where(m > 0, np.where(z > m, np.abs(above(m, z)), np.abs(below(m, z))), 0))

# ---- band + dead: tie x2 ~ x3 (d = x3 - x2) at x2 = x3 = m, third value z = x1 > 0
def window(rows, m, z):
    """Length of {d : r . (z, m, m + d) > 0 for every row r}, elementwise."""
    lo, hi = np.full(np.broadcast(m, z).shape, -np.inf), np.full(np.broadcast(m, z).shape, np.inf)
    for r in rows:
        c0 = r[0] * z + (r[1] + r[2]) * m  # value at d = 0
        c1 = r[2]  # slope in d
        if abs(c1) < 1e-12:
            lo = np.where(c0 > 0, lo, np.inf)
            continue
        cut = -c0 / c1
        if c1 > 0:
            lo = np.maximum(lo, cut)
        else:
            hi = np.minimum(hi, cut)
    return np.clip(hi - lo, 0, None)


def band_rows(path, says=0):
    M, _ = path_matrices(s, parse_path(path))
    return [M[says] - M[k] for k in range(3) if k != says]


def dead_rows(path):
    masks = parse_path(path)
    _, pre = path_matrices(s, masks)
    return [-row for row in pre[2]]  # every step-3 neuron off


x1max = (z > 0) & (m < z)  # x2 = x3 below x1
x1min = (z > 0) & (m > z)
band = np.where(x1max, window(band_rows("1100-0011-1010"), m, z), 0) + \
    np.where(x1min, window(band_rows("1100-1100-0101"), m, z), 0)
dead = np.where(x1max, window(dead_rows("1100-0011-0000"), m, z), 0)
P_band, P_dead = PD0 * expect(band), PD0 * expect(dead)

acc_sym, _, _ = exact_accuracy(s)
acc, _, _ = exact_accuracy(w)
mirror = (1 - acc) - (1 - acc_sym)
pred = 2 * (P_turn + P_step2 + P_band + P_dead)
law = np.sqrt(3) / (2 * np.pi) * (0.7426716 / (0.3693793 * 8.6731361)) ** 2

print("first-order prediction for the trained net (x1 > 0 doubled):")
for name, v in [("turn (x1/x3 boundary shifted)", P_turn), ("step 2 (x1/x2 boundary shifted)", P_step2),
                ("band (x2~x3, mixed path, says x1)", P_band), ("dead (x2~x3, all off, says x1)", P_dead)]:
    print(f"  {name:<36} {2 * v:.4%}")
print(f"  {'total, symmetrized net':<36} {pred:.4%}   exact {1 - acc_sym:.4%}   (miss {pred - (1 - acc_sym):+.4%})")
print(f"  {'+ mirror asymmetry (measured)':<36} {pred + mirror:.4%}   exact {1 - acc:.4%}   (miss {pred + mirror - (1 - acc):+.4%})")
print(f"\nfor comparison, the ideal-design law gives {law:.4%} (miss {law - (1 - acc):+.4%})")
