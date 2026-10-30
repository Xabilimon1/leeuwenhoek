"""Gaussian probability of convex polyhedral cones in R^3, exactly.

A cone {x : n_i . x > 0 for all i} gets probability (solid angle) / (4 pi)
under a standard Gaussian. Cases:
  pointed cone   -> edge rays, fan triangulation, Van Oosterom & Strackee (1983)
  lune (wedge)   -> two antipodal rays; angle of the feasible arc around the axis
  half-space     -> all normals parallel and equal: 1/2
  empty / flat   -> 0
"""

import numpy as np

TOL = 1e-9


def _unit_rows(a):
    a = np.asarray(a, float).reshape(-1, 3)
    norms = np.linalg.norm(a, axis=1)
    a = a[norms > 1e-12] / norms[norms > 1e-12, None]
    return a


def _dedupe(vs):
    out = []
    for v in vs:
        if not any(np.abs(v - q).max() < 1e-7 for q in out):
            out.append(v)
    return np.array(out).reshape(-1, 3)


def _arc_fraction(normals, axis):
    """Fraction of the circle around `axis` where every normal is positive."""
    e1 = np.cross(axis, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(axis, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    theta = np.arctan2(normals @ e2, normals @ e1)
    cuts = np.sort(np.concatenate([theta + np.pi / 2, theta - np.pi / 2]) % (2 * np.pi))
    cuts = np.append(cuts, cuts[0] + 2 * np.pi)
    total = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        mid = (a + b) / 2
        if (np.cos(mid - theta) > 0).all():
            total += b - a
    return total / (2 * np.pi)


def cone_fraction(normals):
    """Gaussian probability of the convex cone {x in R^3 : n . x > 0 for all n}."""
    n = _dedupe(_unit_rows(normals))
    if len(n) == 0:
        return 1.0
    if len(n) == 1:
        return 0.5
    i, j = np.triu_indices(len(n), 1)
    r = np.cross(n[i], n[j])
    keep = np.linalg.norm(r, axis=1) > 1e-12
    if not keep.any():  # all normals parallel; dedupe left opposite pairs only
        return 0.0
    r = r[keep] / np.linalg.norm(r[keep], axis=1, keepdims=True)
    cand = np.concatenate([r, -r])
    rays = _dedupe(cand[(cand @ n.T >= -TOL).all(1)])
    if len(rays) == 0:
        return 0.0
    if len(rays) == 2 and np.abs(rays[0] + rays[1]).max() < 1e-7:
        # a wedge of dihedral angle a covers a / (2 pi) of the sphere
        active = n[np.abs(n @ rays[0]) < 1e-7]
        return _arc_fraction(active, rays[0])
    if len(rays) < 3:
        return 0.0
    centre = rays.mean(0)
    centre /= np.linalg.norm(centre)
    e1 = rays[0] - (rays[0] @ centre) * centre
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    rays = rays[np.argsort(np.arctan2(rays @ e2, rays @ e1))]
    r1, r2, r3 = rays[0], rays[1:-1], rays[2:]
    num = np.abs(np.cross(r2, r3) @ r1)
    den = 1 + r2 @ r1 + np.sum(r2 * r3, axis=1) + r3 @ r1
    return float(2 * np.arctan2(num, den).sum() / (4 * np.pi))
