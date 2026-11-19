"""Time AlgZoo's training step for argmedian h=4 L=3 on this machine (a few thousand steps).

Same model, optimizer, schedule and batch as alg_zoo.training.train; it only
stops early and extrapolates. 2026-11-19 on a MacBook Air M3 (CPU): 0.29 ms
per step with 4 threads and 0.28 ms with 1, so one seed (2^20 steps) takes
about 5 min; the full run of scripts/train_seed_h4.py measured 316 s.

Usage: uv run python scripts/time_training_h4.py
"""

import functools
import time

import torch as th
from alg_zoo.architectures import DistRNN
from alg_zoo.tasks import task_registry
from alg_zoo.training import lr_mult_schedule

th.manual_seed(0)
task, _, _ = task_registry["argmedian"]
model = DistRNN(hidden_size=4, seq_len=3, bias=False)
batch, n_train = 1024, 2**30
n_steps = n_train // batch
opt = th.optim.AdamW(model.parameters(), lr=0.04 / 4**0.5, weight_decay=0.001)
sched = th.optim.lr_scheduler.LambdaLR(opt, functools.partial(
    lr_mult_schedule, n_steps=n_steps, warmup_steps=2**14, decay_min_mult=0.01))
print("torch", th.__version__, "threads", th.get_num_threads(), "steps per seed", n_steps)


def step():
    x = th.randn(batch, 3)
    loss = th.nn.functional.cross_entropy(model(x), task(x))
    loss.backward()
    opt.step()
    sched.step()
    opt.zero_grad()


for threads in (th.get_num_threads(), 1):
    th.set_num_threads(threads)
    for _ in range(200):  # warm-up
        step()
    t, k = time.time(), 3000
    for _ in range(k):
        step()
    dt = (time.time() - t) / k
    print(f"threads {threads}: {dt * 1e3:.2f} ms/step -> one seed (2^30 sequences) {dt * n_steps / 60:.1f} min")
