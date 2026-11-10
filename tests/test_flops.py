import numpy as np

from leeuwenhoek.cones import cone_fraction
from leeuwenhoek.estimate import error_of_cone
from leeuwenhoek.flops import Counter, counted_cone_fraction, estimate_flops, forward_flops
from leeuwenhoek.mechanisms import mechanism_cones
from leeuwenhoek.models import load_argmedian


def test_forward_flops_h4():
    # per step: 4 mul (w_ih x) + 16 mul 12 add (w_hh h) + 4 add + 4 relu = 40
    # readout 12 mul 9 add, argmax 2, true median 3, compare 1
    assert forward_flops(4, 3) == 3 * 40 + 21 + 2 + 3 + 1


def test_counted_cone_fraction_matches_on_mechanism_cones():
    _, w = load_argmedian(4, 3)
    for normals, _ in mechanism_cones(w).values():
        c = Counter()
        assert counted_cone_fraction(normals, c) == cone_fraction(normals)
        assert c.n > 0


def test_counted_cone_fraction_matches_on_random_and_degenerate_cones():
    rng = np.random.default_rng(0)
    cases = [rng.normal(size=(k, 3)) for k in range(1, 8) for _ in range(30)]
    cases += [np.eye(3), [(1, 0, 0), (2, 0, 0)], [(1, 0, 0), (-1, 0, 0)],
              [(1, 0, 0), (0, 1, 0), (1, 1, 0)], [(0, 0, 0)]]
    for normals in cases:
        assert counted_cone_fraction(normals, Counter()) == cone_fraction(normals)


def test_exact_flops_reproduces_the_exact_accuracy():
    from leeuwenhoek.exact import exact_accuracy
    from leeuwenhoek.flops import exact_flops

    _, w = load_argmedian(4, 3)
    acc, flops = exact_flops(w)
    assert acc == exact_accuracy(w)[0]
    assert flops > estimate_flops(w)[1]


def test_estimate_flops_reproduces_the_estimate():
    _, w = load_argmedian(4, 3)
    est, flops, _ = estimate_flops(w)
    ref = 1 - sum(error_of_cone(n, s) for n, s in mechanism_cones(w).values())
    assert est == ref
    assert flops > 0
