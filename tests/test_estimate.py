import numpy as np

from leeuwenhoek.cones import cone_fraction
from leeuwenhoek.estimate import band_cones
from leeuwenhoek.metrics import gaussian_inputs


def test_orthant_is_one_eighth():
    assert np.isclose(cone_fraction(np.eye(3)), 1 / 8)


def test_one_ordering_is_one_sixth():
    # x1 < x2 < x3
    assert np.isclose(cone_fraction([(-1, 1, 0), (0, -1, 1)]), 1 / 6)


def test_band_cones_match_sampling():
    x = gaussian_inputs(4_000_000, 3, seed=3).astype(np.float64)
    for name, (normals, _) in band_cones().items():
        inside = (x @ np.asarray(normals, float).T > 0).all(-1).mean()
        se = np.sqrt(inside / len(x))
        assert abs(cone_fraction(normals) - inside) < 5 * se + 1e-6, name


def test_degenerate_cones():
    assert np.isclose(cone_fraction([(1, 0, 0)]), 1 / 2)  # half-space
    assert np.isclose(cone_fraction([(1, 0, 0), (2, 0, 0)]), 1 / 2)  # repeated plane
    assert cone_fraction([(1, 0, 0), (-1, 0, 0)]) == 0  # empty
    # lune with a redundant third plane: x>0, y>0, x+y>0 -> 1/4
    assert np.isclose(cone_fraction([(1, 0, 0), (0, 1, 0), (1, 1, 0)]), 1 / 4)
    # the six orderings tile the space
    from itertools import permutations
    total = 0
    for a, b, c in permutations(range(3)):
        n1, n2 = np.zeros(3), np.zeros(3)
        n1[[a, b]] = (-1, 1)
        n2[[b, c]] = (-1, 1)
        total += cone_fraction([n1, n2])
    assert np.isclose(total, 1)


def test_estimate_equals_explained_twin_accuracy():
    # the estimate must compute exactly the twin built from the same cones
    from leeuwenhoek.estimate import estimate_accuracy
    from leeuwenhoek.metrics import argmedian_target
    from leeuwenhoek.twins import explained_twin

    x = gaussian_inputs(4_000_000, 3, seed=7).astype(np.float64)
    acc = np.mean(explained_twin(x) == argmedian_target(x))
    est, _ = estimate_accuracy()
    assert abs(acc - est) < 5 * np.sqrt(acc * (1 - acc) / len(x))
