"""Hand-written twin of the 4-unit ReLU RNN trained for argmedian of 3 inputs.

Rule (see EXPLANATION.md): the network returns the true median position,
except inside thin "tie bands" where two inputs are almost equal. The bands
are degree-1 homogeneous (the net has no biases), so every width is a
constant times a magnitude |a| (the tied value) or |c - a| (distance of the
third input from the tie).
"""
import numpy as np

# Band A: x2 ~ x3 (positions 1,2 tied) -> network answers 0.
# Regime "c on the zero side" (t = c/a <= T_IN): band d = x3 - x2 in (lo, hi)
# lo/hi in sign-normalised coordinates (multiply by sign(a)).
A_POS = dict(lo_a=0.0072, lo_g=0.0170, hi_a=0.0164, hi_g=0.0070)   # a > 0
A_NEG = dict(lo_a=0.0128, lo_g=0.0108, hi_a=0.0107, hi_g=0.0135)   # a < 0 (in -x coords)
T_IN = 0.5
# Regime t > T_IN: symmetric half-width |a| * h(t), h piecewise linear through
# the measured knots (t, h).  t = 2 is where the band closes.
KNOTS_T = np.array([0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 6.0])
KNOTS_H = np.array([0.018, 0.036, 0.10, 0.0735, 0.041, 0.0, 0.0125, 0.05])
# Band C: x1 ~ x2 (positions 0,1 tied) -> network answers 1 (true answer 0).
C_IN = 0.0055    # c on zero side: x2 just beyond x1 (away from zero), width C_IN*|a|
C_OUT = 0.0195   # c beyond: x2 just inside x1 (towards zero), width C_OUT*|a|
# Band B: x1 ~ x3 (positions 0,2 tied), c on zero side -> network answers 2.
B_G = 0.0108     # width B_G*|c-a| - B_A*|a| , x3 just beyond x1
B_A = 0.0039


def predict(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    out = np.argsort(x, axis=1)[:, 1].copy()

    # ---- band C: x1 ~ x2, third input x3
    a, d, c = x[:, 0], x[:, 1] - x[:, 0], x[:, 2]
    s = np.where(a >= 0, 1.0, -1.0)
    ds, t = d * s, c * s / np.maximum(np.abs(a), 1e-12)
    inC = ((t < 1) & (ds > 0) & (ds < C_IN * np.abs(a))) | \
          ((t > 1) & (ds < 0) & (ds > -C_OUT * np.abs(a)))
    out[inC] = 1

    # ---- band B: x1 ~ x3, third input x2
    a, d, c = x[:, 0], x[:, 2] - x[:, 0], x[:, 1]
    s = np.where(a >= 0, 1.0, -1.0)
    ds, g = d * s, (a - c) * s
    inB = (g > 0) & (ds > 0) & (ds < B_G * g - B_A * np.abs(a))
    out[inB] = 2

    # ---- band A: x2 ~ x3, third input x1
    a = 0.5 * (x[:, 1] + x[:, 2])
    d = x[:, 2] - x[:, 1]
    c = x[:, 0]
    s = np.where(a >= 0, 1.0, -1.0)
    aa = np.abs(a)
    ds, t = d * s, c * s / np.maximum(aa, 1e-12)
    g = np.abs(c - a)
    pos = s > 0
    lo = np.where(pos, -(A_POS['lo_a'] * aa + A_POS['lo_g'] * g),
                  -(A_NEG['lo_a'] * aa + A_NEG['lo_g'] * g))
    hi = np.where(pos, A_POS['hi_a'] * aa + A_POS['hi_g'] * g,
                  A_NEG['hi_a'] * aa + A_NEG['hi_g'] * g)
    in_low = (t <= T_IN) & (ds > lo) & (ds < hi)
    hw = aa * np.interp(t, KNOTS_T, KNOTS_H, right=KNOTS_H[-1] / KNOTS_T[-1] * 1.0)
    hw = np.where(t > KNOTS_T[-1], aa * 0.0125 * (t - 2.0), hw)
    in_high = (t > T_IN) & (np.abs(ds) < hw)
    out[in_low | in_high] = 0
    return out
