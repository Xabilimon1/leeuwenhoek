"""Inside one ReLU activation path the RNN is linear; compute that linear map."""

import numpy as np


def parse_path(path):
    """'1100-0011-1010' -> list of 0/1 masks, one per step."""
    return [np.array([int(b) for b in p], float) for p in path.split("-")]


def path_matrices(w, masks):
    """Linear maps for a path. Returns (M, pre) with logits = M @ x and
    pre[t] (h x L) the pre-activations at step t as linear functions of x."""
    w_ih, w_hh, w_out = (np.asarray(w[k], np.float64) for k in ("w_ih", "w_hh", "w_out"))
    L = len(masks)
    h = np.zeros((len(w_ih), L))  # state as a linear function of x: h = S @ x
    pre = []
    for t, d in enumerate(masks):
        z = w_hh @ h
        z[:, t] += w_ih
        pre.append(z)
        h = np.diag(d) @ z
    return w_out @ h, pre
