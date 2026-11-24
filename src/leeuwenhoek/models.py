"""Load AlgZoo models, caching the weights locally so we only download once."""

from pathlib import Path

import torch as th
from alg_zoo import zoo_argmedian
from alg_zoo.architectures import DistRNN

CACHE = Path(__file__).resolve().parents[2] / "data" / "weights"


def load_argmedian(hidden_size, seq_len, *, seed=None, n_seqs=None):
    """Return (torch model, numpy weights dict) for an argmedian DistRNN."""
    tag_seqs = "final" if n_seqs is None else n_seqs
    tag_seed = "best" if seed is None else seed
    path = CACHE / f"argmedian_{hidden_size}_{seq_len}_{tag_seqs}_{tag_seed}.pth"
    if path.exists():
        model = DistRNN(hidden_size=hidden_size, seq_len=seq_len)
        model.load_state_dict(th.load(path, weights_only=True))
    else:
        model = zoo_argmedian(hidden_size, seq_len, seed=seed, n_seqs=n_seqs)
        CACHE.mkdir(parents=True, exist_ok=True)
        th.save(model.state_dict(), path)
    model.eval()
    return model, numpy_weights(model)


def numpy_weights(model):
    sd = model.state_dict()
    return {
        "w_ih": sd["rnn.weight_ih_l0"].numpy()[:, 0].copy(),  # (h,)
        "w_hh": sd["rnn.weight_hh_l0"].numpy().copy(),  # (h, h)
        "w_out": sd["linear.weight"].numpy().copy(),  # (L, h)
    }


def mirror_init_(model, pairs=((0, 3), (1, 2))):
    """Make a DistRNN exactly sign-mirror symmetric, in place, keeping the first half.

    Neurons pairs[k] = (i, j): j becomes the mirror of i, so that with P the swap
    P w_ih = -w_ih, P W_hh P^T = W_hh and W_out P^T = W_out (structure.mirror_permutation
    gives residual 0). Only copies existing values: it draws no random numbers, so
    the data stream of a training run is the same with or without it.
    """
    sd = model.state_dict()
    wi, wh, wo = sd["rnn.weight_ih_l0"], sd["rnn.weight_hh_l0"], sd["linear.weight"]
    perm = list(range(wi.shape[0]))
    for i, j in pairs:
        perm[i], perm[j] = j, i
    with th.no_grad():
        for i, _ in pairs:
            wi[perm[i], 0] = -wi[i, 0]
            for k in range(wi.shape[0]):
                wh[perm[i], perm[k]] = wh[i, k]
            wo[:, perm[i]] = wo[:, i]
    return model
