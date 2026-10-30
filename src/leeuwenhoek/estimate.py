"""Mechanistic accuracy estimate for argmedian h=4 L=3, with no sampling.

Every error region we explained is a polyhedral cone {x : n_i . x > 0 for all i}.
A Gaussian puts probability (solid angle) / (4 pi) on a cone, computed
exactly in cones.py.
"""

from leeuwenhoek.cones import cone_fraction


def band_cones():
    """The explained error cones, written for x1 > 0 (mirror images double them).

    Coefficients are the ones derived from the weights in tie_band.py and
    tie_band_step2.py (see twins.algorithmic_twin_bands).
    """
    x1_pos = (1, 0, 0)
    return {
        # step-3 band, x2 < x1: -0.0104 + 0.0237 v < delta < 0.0138 - 0.0240 v
        "step-3 band, x1 max": [x1_pos, (1, -1, 0), (0.0104, -1.0237, 1), (0.0138, 1 - 0.0240, -1)],
        # step-3 band, x2 > x1: 0.0054 - 0.0240 v < delta < -0.0196 + 0.0237 v
        "step-3 band, x1 min": [x1_pos, (-1, 1, 0), (-0.0054, -(1 - 0.0240), 1), (-0.0196, 1.0237, -1)],
        # step-2 sliver, x3 above: -0.0213 + 0.0018 r < e < 0
        "step-2 sliver, x3 above": [x1_pos, (-1, 0, 1), (0, -1, 1), (-1 + 0.0213, 1, -0.0018), (1, -1, 0)],
        # step-2 sliver, x3 below: 0 < e < 0.0053 + 0.0008 r
        "step-2 sliver, x3 below": [x1_pos, (1, 0, -1), (0, 1, -1), (-1, 1, 0), (1.0053, -1, 0.0008)],
    }


def estimate_accuracy():
    parts = {k: 2 * cone_fraction(v) for k, v in band_cones().items()}
    return 1 - sum(parts.values()), parts
