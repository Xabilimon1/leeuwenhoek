"""The designed RNN must do what its derivation claims."""

import numpy as np

from leeuwenhoek.design import B, designed_weights, derived
from leeuwenhoek.forward import forward
from leeuwenhoek.linear import parse_path, path_matrices
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs


def test_comparison_channel_is_exact_on_clean_paths():
    a, b, e = 8.67, 0.74, 0.37
    w = designed_weights(a, b, e, dtype=np.float64)
    for path in ["1100-0011-1100", "1100-1100-0011", "1100-0011-0011", "1100-1100-1100"]:
        masks = parse_path(path)
        _, pre = path_matrices(w, masks)
        for t in (1, 2):
            s_pos = (B @ pre[t])[0]  # s+ pre-activation as a linear form in x
            want = np.zeros(3)
            want[t], want[t - 1] = a, -a
            np.testing.assert_allclose(s_pos, want, atol=1e-9)


def test_readout_is_the_median_rule_on_clean_paths():
    a, b, e = 8.67, 0.74, 0.37
    h = derived(a, b, e)["h"]
    w = designed_weights(a, b, e, B2=1.0, B3=1.0, dtype=np.float64)
    # path +-+ : x1 > 0, x2 < x1, x3 > x2 (positive half at the end, sign +1)
    M, _ = path_matrices(w, parse_path("1100-0011-1100"))
    np.testing.assert_allclose(M[1] - M[0], h * np.array([-1, 1, 0]), atol=1e-9)
    np.testing.assert_allclose(M[0] - M[2], h * np.array([-1, 0, 1]), atol=1e-9)


def test_designed_rnn_is_almost_perfect_away_from_ties():
    w = designed_weights()
    x = gaussian_inputs(200_000, 3, seed=11)
    gaps = np.abs(np.diff(np.sort(x, -1), axis=-1)).min(-1) / np.abs(x).max(-1)
    far = gaps > 0.3
    pred = forward(w, x).argmax(-1)
    assert np.mean(pred[far] == argmedian_target(x)[far]) > 0.999
