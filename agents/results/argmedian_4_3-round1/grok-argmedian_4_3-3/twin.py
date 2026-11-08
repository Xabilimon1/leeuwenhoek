"""Hand rule for the median-position network.

The network returns the median's index, except in a few thin regions derived
in EXPLANATION.md: an abstention window around z ≈ y (answer 0), two shifted
cuts between x and z, two shifted cuts between x and y, an early flip to
index 2 when y is only slightly larger than x, and a dead zone in which
every ReLU is off (answer 0). Each constant below is one of those cuts.
"""

import numpy as np

# Abstention window: logit 0 beats logit 2 below and logit 1 above.
# z is between (A_LO * x + LO * y) and (A_HI * x + HI * y).
LO = 0.97595
HI = 1.02373
A_LO = 0.0103
A_HI = -0.0143

# y much smaller than x (negative-pair regime). The x-versus-z cut is
# z = XZ_SMALL_X * x + XZ_SMALL_Y * y, not z = x.
GATE_SMALL = 0.78820
XZ_SMALL_X = 1.00692
XZ_SMALL_Y = -0.01090

# y much larger than x (positive-pair regime).
GATE_LARGE = 1.26817
XZ_LARGE_X = 1.02251
XZ_LARGE_Y = -0.00575

# Comparable y (mixed units). Cuts between answer 0 and answer 1.
XY_BELOW_X = 0.97850
XY_BELOW_Z = 0.00194
XY_ABOVE_X = 1.00433
XY_ABOVE_Z = 0.00078

# Same regime, cut between answer 0 and answer 2, on the side z < x < y.
MID_HI = 1.23916
MID_Z_X = 0.8560
MID_Z_Y = 0.0769

# Dead zone: y/x inside the two step-2 gates and z between all four step-3 gates.
GATE_MID_LO = 0.8073
GATE_MID_HI = 1.2392
DEAD_U0 = (0.2433, 0.7121)
DEAD_U1 = (0.5263, 0.4235)
DEAD_U2 = (-0.2430, 1.1704)
DEAD_U3 = (-0.8672, 1.7914)


def _median_index(x):
    return np.argsort(x, axis=1)[:, 1]


def _abstain(med, a, b, c):
    """Answer 0 when z lies between the two readout edges around y."""
    edge_lo = A_LO * a + LO * b
    edge_hi = A_HI * a + HI * b
    lo = np.minimum(edge_lo, edge_hi)
    hi = np.maximum(edge_lo, edge_hi)
    med[(c > lo) & (c < hi)] = 0


def _wedges(med, a, b, c):
    """Shifted cuts. Written for the a>0 chart; call again on negated inputs."""
    # y << x: answer 2 between z = x and the shifted cut.
    line = XZ_SMALL_X * a + XZ_SMALL_Y * b
    med[(a > 0) & (b < GATE_SMALL * a) & (c > a) & (c < line)] = 2

    # y >> x: 0 on the low side of the cut, 2 on the high side.
    line = XZ_LARGE_X * a + XZ_LARGE_Y * b
    lo = np.minimum(a, line)
    hi = np.maximum(a, line)
    wedge = (a > 0) & (b > GATE_LARGE * a) & (c > lo) & (c < hi) & (c < b)
    med[wedge] = np.where(c[wedge] < line[wedge], 0, 2)

    # Comparable y, z largest: answer 1 once y clears a cut just below x.
    line = XY_BELOW_X * a + XY_BELOW_Z * c
    med[(a > 0) & (c > a) & (b > line) & (b < a)] = 1

    # Comparable y, z smallest: answer 1 until y clears a cut just above x.
    line = XY_ABOVE_X * a + XY_ABOVE_Z * c
    med[(a > 0) & (c < a) & (b > a) & (b < line)] = 1

    # y only slightly above x: flip to 2 before z reaches x.
    line = MID_Z_X * a + MID_Z_Y * b
    med[(a > 0) & (b > a) & (b < MID_HI * a) & (c > line) & (c < a)] = 2


def _dead(med, a, b, c):
    """All four step-3 gates off -> logits are 0 -> argmax is 0."""
    upper = np.minimum(
        DEAD_U0[0] * a + DEAD_U0[1] * b, DEAD_U1[0] * a + DEAD_U1[1] * b
    )
    lower = np.maximum(
        DEAD_U2[0] * a + DEAD_U2[1] * b, DEAD_U3[0] * a + DEAD_U3[1] * b
    )
    dead = (
        (a > 0)
        & (b > GATE_MID_LO * a)
        & (b < GATE_MID_HI * a)
        & (c < upper)
        & (c > lower)
    )
    med[dead] = 0


def predict(x: np.ndarray) -> np.ndarray:
    """Median index, with the exception regions in EXPLANATION.md.

    x: (n, 3) float -> (n,) ints in 0..2
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x must have shape (n, 3)")
    med = _median_index(x).copy()
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    # The window scales with y, so one pass covers both signs.
    _abstain(med, a, b, c)
    # Cuts are derived on x>0. Negating every input keeps the median index,
    # and the weight mirror sends that chart onto x<0, up to a small
    # training residual.
    _wedges(med, a, b, c)
    _wedges(med, -a, -b, -c)
    _dead(med, a, b, c)
    _dead(med, -a, -b, -c)
    return med.astype(np.int64, copy=False)
