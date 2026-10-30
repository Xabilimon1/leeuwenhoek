"""Checks behind the sphere map: homogeneity and the equal-area claim."""

import numpy as np

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import accuracy, argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.sphere import directions, grid


def test_model_is_positively_homogeneous():
    # float64: in float32 the logits at c=100 (~1e4) carry rounding of ~1e-3
    _, w = load_argmedian(4, 3)
    w = {k: v.astype(np.float64) for k, v in w.items()}
    x = gaussian_inputs(10_000, 3).astype(np.float64)
    for c in [0.01, 0.5, 3.0, 100.0]:
        np.testing.assert_allclose(forward(w, c * x), c * forward(w, x), rtol=1e-9, atol=1e-9)


def test_directions_are_unit_vectors():
    theta, z = grid(64, 32)
    np.testing.assert_allclose(np.linalg.norm(directions(theta, z), axis=-1), 1, atol=1e-6)


def test_grid_accuracy_matches_sampling():
    # If the map is equal-area, the fraction of correct pixels must match
    # Gaussian sampling up to sampling noise.
    _, w = load_argmedian(4, 3)
    u = directions(*grid(720, 360)).reshape(-1, 3)
    on_grid = accuracy(forward(w, u), argmedian_target(u))

    x = gaussian_inputs(1_000_000, 3)
    sampled = accuracy(forward(w, x), argmedian_target(x))
    se = np.sqrt(sampled * (1 - sampled) / len(x))
    assert abs(on_grid - sampled) < 5 * se
