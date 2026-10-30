"""Exact accuracy from the weights must agree with sampling, for several models."""

from collections import Counter

import numpy as np
import pytest

from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target, gaussian_inputs
from leeuwenhoek.models import load_argmedian

X = gaussian_inputs(2_000_000, 3, seed=6)


@pytest.mark.parametrize("hidden_size", [2, 4, 8])
def test_exact_accuracy_matches_sampling(hidden_size):
    _, w = load_argmedian(hidden_size, 3)
    acc, total, _ = exact_accuracy(w)
    assert abs(total - 1) < 1e-6
    sampled = np.mean(forward(w, X).argmax(-1) == argmedian_target(X))
    se = np.sqrt(sampled * (1 - sampled) / len(X))
    assert abs(acc - sampled) < 5 * se


def test_path_probabilities_match_sampling():
    _, w = load_argmedian(4, 3)
    _, _, pieces = exact_accuracy(w)
    exact = Counter()
    for p in pieces:
        exact[p["path"]] += p["prob"]
    _, states = forward(w, X, return_states=True)
    keys = ["-".join("".join("1" if b else "0" for b in s) for s in row) for row in states > 0]
    sampled = Counter(keys)
    for path, p in exact.most_common(10):
        f = sampled[path] / len(X)
        assert abs(p - f) < 5 * np.sqrt(p / len(X)), path
