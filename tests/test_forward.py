"""Our hand-written numpy forward must reproduce alg-zoo's torch model exactly.

This comparator is the same one the handcrafted twin will have to pass later.
"""

import numpy as np
import pytest
import torch as th

from leeuwenhoek.forward import forward
from leeuwenhoek.models import load_argmedian


@pytest.mark.parametrize("hidden_size", [2, 4, 8])
def test_numpy_forward_matches_torch(hidden_size):
    model, weights = load_argmedian(hidden_size, 3)
    x = np.random.default_rng(0).standard_normal((10_000, 3)).astype(np.float32)

    with th.no_grad():
        expected = model(th.from_numpy(x)).numpy()
    got = forward(weights, x)

    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert (got.argmax(-1) == expected.argmax(-1)).all()
