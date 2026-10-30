"""Weights for argmedian L=3 built from the explanation, not from training.

Each mirror half (n1, n2) / (n4, n3) carries two channels, written with a sign
so both halves follow one rule (sign = +1 on the positive half, -1 on the
negative half):
    sigma_t = a (x_t - x_{t-1})          comparison; its sign picks the half
    mu_t    = b x_t + c sigma_{t-1} - e mu_{t-1}     memory
The recurrence rebuilds the previous value as  x_{t-1} = (p sigma + q mu) / a,
so sigma_{t+1} = a x_{t+1} - p sigma_t - q mu_t.

Requirements and what they force:
  (1) sigma_2 exact:            p a + q b = a
  (2) sigma_3 exact (no x1):    p a = q g,           g = c a - e b
  (3) readout can compare x1 with x3 on turns AND x1 with x2 on monotone runs
      using only (sigma_3, mu_3):  mu_3 = b x3 + g x2 - h x1 with h = b + g,
      which fixes c = b (1 - e + e^2) / (e a)
So a, b, e are free (a only sets the scale) and c, p, q follow.

Readout: logit_k = alpha_k * s + beta_k * d on whichever half is on, with
logit x1 = 0, beta_x2 = B2, beta_x3 = -B3, and alphas chosen so that
    L_x2 - L_x1 = B2 h (x2 - x1)  and  L_x1 - L_x3 = B3 h (x3 - x1)   (times the sign)
which is exactly the median rule on every clean path.
Neurons are n1 = (s - d)/2, n2 = (s + d)/2 (and mirrored): both are on only
while |sigma| > |mu|, so near ties the pair breaks. That is where it fails.
"""

import numpy as np

# basis change (n1, n2, n3, n4) -> (s+, d+, s-, d-), same as scripts/memory_basis.py
B = np.array([[1, 1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]], float)
BINV = np.linalg.inv(B)


def derived(a, b, e):
    c = b * (1 - e + e**2) / (e * a)
    g = c * a - e * b
    q = a / (g + b)
    p = q * g / a
    return {"c": c, "g": g, "h": b + g, "p": p, "q": q}


def read_off(w):
    """The design's free numbers as the trained net has them (mirror-symmetrized).

    e differs slightly between same-half and cross-half transitions in the
    trained net; the design uses one e, the mean of the two.
    """
    from leeuwenhoek.twins import symmetrize

    s = symmetrize(w)
    W = B @ s["w_hh"].astype(np.float64) @ BINV
    w_ih_sd = B @ s["w_ih"].astype(np.float64)
    rel = s["w_out"].astype(np.float64) @ BINV
    rel = rel - rel[0]  # relative to the x1 logit: a common offset changes nothing
    return {
        "a": float(w_ih_sd[0]),
        "b": float(w_ih_sd[1]),
        "e": float((-W[1, 1] + W[1, 3]) / 2),
        "e_same": float(-W[1, 1]),
        "e_cross": float(W[1, 3]),
        "B2": float(rel[1, 1]),
        "B3": float(-rel[2, 1]),
    }


def designed_weights(a=8.67, b=0.74, e=0.37, B2=15.7, B3=15.2, dtype=np.float32):
    k = derived(a, b, e)
    p, q, c = k["p"], k["q"], k["c"]
    same = np.array([[-p, -q], [c, -e]])  # previous step on the same half
    cross = np.array([[p, q], [-c, e]])  # previous step on the other half
    w_sd = np.block([[same, cross], [cross, same]])
    w_ih_sd = np.array([a, b, -a, -b])
    # readout in (s, d) of the active half, identical for both halves
    beta = np.array([0.0, B2, -B3])
    alpha = np.array([0.0, -B2 * b / a, -B3 * k["g"] / a])
    u_sd = np.stack([alpha, beta, alpha, beta], axis=1)
    return {
        "w_ih": (BINV @ w_ih_sd).astype(dtype),
        "w_hh": (BINV @ w_sd @ B).astype(dtype),
        "w_out": (u_sd @ B).astype(dtype),
    }
