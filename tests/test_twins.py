import numpy as np

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.twins import P, algorithmic_twin, symmetrize


def test_clean_algorithm_is_exactly_argmedian():
    x = gaussian_inputs(100_000, 3)
    assert (algorithmic_twin(x) == argmedian_target(x)).all()


def test_symmetrized_weights_are_exactly_mirror_symmetric():
    _, w = load_argmedian(4, 3)
    s = symmetrize(w)
    np.testing.assert_allclose(s["w_ih"], -P @ s["w_ih"], atol=1e-6)
    np.testing.assert_allclose(s["w_hh"], P @ s["w_hh"] @ P, atol=1e-6)
    np.testing.assert_allclose(s["w_out"], s["w_out"] @ P, atol=1e-6)


def test_exact_symmetry_gives_identical_answers_for_x_and_minus_x():
    _, w = load_argmedian(4, 3)
    s = {k: v.astype(np.float64) for k, v in symmetrize(w).items()}
    x = gaussian_inputs(100_000, 3).astype(np.float64)
    np.testing.assert_allclose(forward(s, -x), forward(s, x), atol=1e-9)
