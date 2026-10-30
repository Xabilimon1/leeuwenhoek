"""Mechanistic accuracy estimate for argmedian h=4 L=3, with no sampling.

Every error region we explained is a polyhedral cone {x : n_i . x > 0 for all i}.
A Gaussian puts probability (solid angle) / (4 pi) on a cone, and the solid
angle of a convex polyhedral cone is exact: find its edge rays, fan-triangulate,
and add the triangles (Van Oosterom & Strackee 1983).
"""

from itertools import combinations

import numpy as np


def cone_fraction(normals, tol=1e-12):
    """Gaussian probability of the convex cone {x in R^3 : n . x > 0 for all n}."""
    normals = np.asarray(normals, float)
    if len(normals) == 2:
        # A lune (wedge) between two planes: no corner rays, angle = pi - angle(n1, n2)
        n1, n2 = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return (np.pi - np.arccos(np.clip(n1 @ n2, -1, 1))) / (2 * np.pi)
    rays = []
    for a, b in combinations(normals, 2):
        r = np.cross(a, b)
        if np.linalg.norm(r) < tol:
            continue
        r /= np.linalg.norm(r)
        for s in (r, -r):
            if (normals @ s >= -1e-9).all() and not any(np.allclose(s, q) for q in rays):
                rays.append(s)
    if len(rays) == 2 and np.allclose(rays[0], -rays[1]):
        raise ValueError("lune with redundant planes: pass only its two bounding planes")
    if len(rays) < 3:
        return 0.0
    rays = np.array(rays)
    centre = rays.mean(0)
    centre /= np.linalg.norm(centre)
    # order rays by angle around the centre
    e1 = rays[0] - (rays[0] @ centre) * centre
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    rays = rays[np.argsort(np.arctan2(rays @ e2, rays @ e1))]
    omega = 0.0
    for i in range(1, len(rays) - 1):
        r1, r2, r3 = rays[0], rays[i], rays[i + 1]
        num = abs(r1 @ np.cross(r2, r3))
        den = 1 + r1 @ r2 + r2 @ r3 + r3 @ r1
        omega += 2 * np.arctan2(num, den)
    return omega / (4 * np.pi)


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
