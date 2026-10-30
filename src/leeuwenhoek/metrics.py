"""Sampling-based measurements: accuracy and entry-by-entry agreement."""

import numpy as np


def argmedian_target(x):
    # Same definition as alg_zoo.tasks.task_argmedian
    return np.argsort(x, axis=-1)[..., x.shape[-1] // 2]


def gaussian_inputs(n, seq_len, seed=0):
    return np.random.default_rng(seed).standard_normal((n, seq_len)).astype(np.float32)


def accuracy(logits, targets):
    return float((logits.argmax(-1) == targets).mean())


def agreement(logits_a, logits_b):
    """Fraction of inputs where two models make the same prediction."""
    return float((logits_a.argmax(-1) == logits_b.argmax(-1)).mean())
