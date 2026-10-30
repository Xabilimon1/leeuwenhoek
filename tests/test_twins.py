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


def test_mechanism_twin_agreement_and_estimate():
    from leeuwenhoek.estimate import error_of_cone
    from leeuwenhoek.mechanisms import mechanism_cones
    from leeuwenhoek.metrics import argmedian_target
    from leeuwenhoek.twins import twin_from_cones

    _, w = load_argmedian(4, 3)
    cones = mechanism_cones(w)
    x = gaussian_inputs(1_000_000, 3, seed=10)
    twin = twin_from_cones(x, cones)
    assert np.mean(twin == forward(w, x).argmax(-1)) > 0.998
    # the estimate is the exact accuracy of this twin
    acc = np.mean(twin == argmedian_target(x))
    est = 1 - sum(error_of_cone(n, s) for n, s in cones.values())
    assert abs(acc - est) < 5 * np.sqrt(acc * (1 - acc) / len(x))
