"""Load the network and run it. Needs numpy only.

    h_0 = 0
    h_t = relu(w_ih * x_t + w_hh @ h_{t-1})    for t = 1..L
    logits = w_out @ h_L

Every call to run_model (and predict) appends one line to model_calls.jsonl,
so the number of model evaluations can be counted afterwards.
"""

import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOG = HERE / "model_calls.jsonl"


def load_weights():
    """Return a dict with w_ih (h,), w_hh (h, h) and w_out (L, h), float64."""
    with np.load(HERE / "weights.npz") as f:
        return {k: f[k].astype(np.float64) for k in ("w_ih", "w_hh", "w_out")}


def run_model(x, note=""):
    """Logits of shape (n, L) for inputs x of shape (n, L)."""
    w = load_weights()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    h = np.zeros((x.shape[0], w["w_hh"].shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(0.0, x[:, t, None] * w["w_ih"] + h @ w["w_hh"].T)
    with open(LOG, "a") as f:
        f.write(json.dumps({"time": time.time(), "n_inputs": int(x.shape[0]), "note": note}) + "\n")
    return h @ w["w_out"].T


def predict(x, note=""):
    """Predicted positions (n,), the argmax of the logits."""
    return run_model(x, note).argmax(-1)
