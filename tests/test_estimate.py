import numpy as np

from leeuwenhoek.estimate import band_cones, cone_fraction
from leeuwenhoek.metrics import gaussian_inputs


def test_orthant_is_one_eighth():
    assert np.isclose(cone_fraction(np.eye(3)), 1 / 8)


def test_one_ordering_is_one_sixth():
    # x1 < x2 < x3
    assert np.isclose(cone_fraction([(-1, 1, 0), (0, -1, 1)]), 1 / 6)


def test_band_cones_match_sampling():
    x = gaussian_inputs(4_000_000, 3, seed=3).astype(np.float64)
    for name, normals in band_cones().items():
        inside = (x @ np.asarray(normals, float).T > 0).all(-1).mean()
        se = np.sqrt(inside / len(x))
        assert abs(cone_fraction(normals) - inside) < 5 * se + 1e-6, name
