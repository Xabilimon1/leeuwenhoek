"""Can AlgZoo's single seed be split into an init seed and a data seed? Tiny runs (2^16 sequences)."""
import os
import sys
import tempfile
from pathlib import Path

import torch as th
import alg_zoo.training as T

N = 2**16  # 64 steps of 1024: seconds


def run(seed, data_seed=None):
    d = tempfile.mkdtemp()
    os.environ["RESULTS_DIR"] = d
    orig = T.save_model
    first = {"done": False}

    def save_and_reseed(model, name):
        orig(model, name)
        if not first["done"]:  # the step-0 save happens right after the model is built, before any data
            first["done"] = True
            if data_seed is not None:
                th.manual_seed(data_seed)

    T.save_model = save_and_reseed
    th.set_num_threads(1)
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            T.train("argmedian", hidden_size=4, seq_len=3, seed=seed, n_train=N, n_val=1024, checkpoints=2, log_every=2**20)
    finally:
        T.save_model = orig
    sd = lambda n: th.load(Path(d) / f"argmedian_4_3_{n}_{seed}.pth", weights_only=True)  # noqa: E731
    return sd(0), sd(N)


def same(a, b):
    return all(th.equal(a[k], b[k]) for k in a)


i_a, f_a = run(7)
i_b, f_b = run(7)
print("1. same seed twice, bit-identical final (CPU determinism):", same(f_a, f_b))
i_c, f_c = run(7, data_seed=None)
print("2. patch without data seed reproduces the original:", same(i_a, i_c) and same(f_a, f_c))
i_d, f_d = run(7, data_seed=1000)
print("3. init 7 + data 1000: same initial weights as seed 7:", same(i_a, i_d), "| final differs:", not same(f_a, f_d))
i_e, f_e = run(8, data_seed=1000)
print("4. init 8 + data 1000: different initial weights from seed 7:", not same(i_a, i_e))
i_f, f_f = run(7, data_seed=1000)
print("5. init 7 + data 1000 twice, bit-identical:", same(f_d, f_f))
