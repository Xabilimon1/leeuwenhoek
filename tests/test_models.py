import numpy as np
import torch as th
from alg_zoo.architectures import DistRNN

from leeuwenhoek.models import mirror_init_, numpy_weights
from leeuwenhoek.structure import mirror_permutation


def test_mirror_init_is_exact_mirror_and_keeps_first_half():
    th.manual_seed(3)
    m = DistRNN(hidden_size=4, seq_len=3)
    before = numpy_weights(m)
    after = numpy_weights(mirror_init_(m))
    perm, res = mirror_permutation(after)
    assert res < 1e-6
    assert tuple(perm) == (3, 2, 1, 0)
    assert np.array_equal(after["w_ih"][:2], before["w_ih"][:2])
    assert np.array_equal(after["w_hh"][:2], before["w_hh"][:2])
    assert np.array_equal(after["w_out"][:, :2], before["w_out"][:, :2])


def test_mirror_init_draws_no_random_numbers():
    th.manual_seed(5)
    DistRNN(hidden_size=4, seq_len=3)
    plain = th.randn(3)
    th.manual_seed(5)
    mirror_init_(DistRNN(hidden_size=4, seq_len=3))
    assert th.equal(th.randn(3), plain)
