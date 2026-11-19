"""Structure diagnostics must reproduce what we already know about seed 0."""

import numpy as np

from leeuwenhoek.design import designed_weights
from leeuwenhoek.linear import parse_path, path_matrices
from leeuwenhoek.metrics import gaussian_inputs
from leeuwenhoek.models import load_argmedian
from leeuwenhoek.structure import (
    best_design_agreement,
    comparison_from_patterns,
    error_profile,
    mirror_permutation,
    neuron_boundaries,
    pattern_keys,
    state_singular_values,
    symmetry_agreement,
)

_, W0 = load_argmedian(4, 3, seed=0)
X = gaussian_inputs(200_000, 3, seed=5)


def test_error_profile_matches_the_seed_0_note():
    p = error_profile(W0)
    assert abs(p["accuracy"] - 0.9845504) < 1e-6
    assert abs(p["wrong_share"][0] - 0.768) < 0.002  # 77 % of wrong answers are x1
    assert abs(p["dead_share"] - 0.108) < 0.002  # dead state = 10.8 % of the errors
    assert abs(sum(p["error_by_median"]) - p["error"]) < 1e-9


def test_mirror_permutation_finds_the_seed_0_mirror_and_the_designed_one():
    perm, res = mirror_permutation(W0)
    assert perm == (3, 2, 1, 0)
    assert res < 0.01
    perm, res = mirror_permutation(designed_weights(dtype=np.float64))
    assert perm == (3, 2, 1, 0)
    assert res < 1e-12


def test_mirror_permutation_reports_no_mirror_for_an_asymmetric_net():
    w = dict(W0, w_ih=np.array([1.0, 2.0, 3.0, 4.0], np.float32))
    _, res = mirror_permutation(w)
    assert res > 0.5


def test_symmetry_agreement_is_exact_for_designed_weights():
    assert symmetry_agreement(designed_weights(dtype=np.float64), X.astype(np.float64)) == 1.0
    assert symmetry_agreement(W0, X) > 0.99


def test_comparison_from_patterns_scores_lookup_tables():
    c = X[:, 1] > X[:, 0]
    keys = np.where(c, "a", "b")
    assert comparison_from_patterns(keys, c) == 1.0
    keys = np.where(X[:, 2] > 0, "a", "b")  # independent of the comparison
    assert abs(comparison_from_patterns(keys, c) - 0.5) < 0.01


def test_pattern_keys_round_trip_with_path_matrices():
    from leeuwenhoek.forward import forward

    _, states = forward(W0, X[:10], return_states=True)
    keys = pattern_keys(states)
    for i, k in enumerate(keys):
        M, _ = path_matrices(W0, parse_path(k))
        np.testing.assert_allclose(M @ X[i].astype(np.float64), forward(W0, X[i : i + 1])[0], atol=1e-3)


def test_neuron_boundaries_give_the_step_2_thresholds_of_the_note():
    # seed 0, x1 > 0 (step 1 = 1100): neuron 3 is on if x2 < 0.79 x1, neuron 4 if x2 < 1.24 x1
    b = neuron_boundaries(W0, parse_path("1100"))
    assert b[2]["side"] == "below" and abs(b[2]["ref"][0] - 0.79) < 0.01
    assert b[3]["side"] == "below" and abs(b[3]["ref"][0] - 1.24) < 0.01
    # neurons 1, 2 are the mirror: on if x2 above a threshold
    assert b[0]["side"] == "above" and b[1]["side"] == "above"


def test_state_singular_values_show_the_two_channels_of_seed_0():
    sv = state_singular_values(W0, parse_path("1100-0011"))
    assert sv.shape == (2,)
    assert 8 < sv[0] < 9.5 and 0.8 < sv[1] < 1.2  # comparison ~ a = 8.67, memory ~ b = 0.74..1
    assert state_singular_values(W0, parse_path("1100-0000")).max() == 0


def test_best_design_agreement_describes_seed_0_but_not_an_arbitrary_net():
    best = best_design_agreement(W0, X[:50_000], ratios=(0.05, 0.086, 0.15), es=(0.3, 0.37, 0.5))
    assert best["agreement"] > 0.98
    assert abs(best["b_over_a"] - 0.086) < 1e-9  # the net's own b/a wins; e is less sharp
    _, w4 = load_argmedian(4, 3, seed=2)
    assert best_design_agreement(w4, X[:50_000], ratios=(0.086,), es=(0.37,))["agreement"] < 0.9
