"""Hand-written forward pass of a bias-free ReLU DistRNN, in numpy.

Convention, checked against alg_zoo/architectures.py (torch.nn.RNN, relu, no bias):
    h_0 = 0
    h_t = relu(w_ih * x_t + w_hh @ h_{t-1})    for t = 1..L
    logits = w_out @ h_L                        (only the final state is read)
"""

import numpy as np


def forward(weights, x, return_states=False):
    """x: (n, L) inputs. Returns logits (n, L), and states (n, L, h) if asked."""
    w_ih, w_hh, w_out = weights["w_ih"], weights["w_hh"], weights["w_out"]
    h = np.zeros((x.shape[0], w_hh.shape[0]), dtype=x.dtype)
    states = []
    for t in range(x.shape[1]):
        h = np.maximum(0, x[:, t, None] * w_ih + h @ w_hh.T)
        states.append(h)
    logits = h @ w_out.T
    if return_states:
        return logits, np.stack(states, axis=1)
    return logits
