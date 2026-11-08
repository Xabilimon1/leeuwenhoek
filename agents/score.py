"""Score one run directory and print JSON.

    uv run python -m agents.score ~/agent-runs/claude-argmedian_4_3-1 [--n 1000000]

- twin: agreement of twin.predict with the real model on N fixed-seed
  Gaussian inputs (the model's answers, mistakes included).
- cheating: a static scan of the agent's .py files for ways of reading the
  weights, a count of weight values pasted into twin.py, and a dynamic check:
  the twin is run from a scratch directory holding only the agent's .py files
  (no weights.npz, no load_model.py). A twin that needs the weights fails there.
- estimate: first number in estimate.txt (a fraction, or a percentage if
  > 1) against the exact accuracy stored in the manifest next to the run.
- weights_intact: the weights and loader still match the manifest hashes.
- model_calls: evaluations logged in model_calls.jsonl.

The twin is agent-written code and gets executed: run this only on runs you
have looked at.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from agents.make_run import sha256
from leeuwenhoek.forward import forward

SEED = 20261108
STATIC_PATTERNS = {
    "weights file": r"weights\.npz|\.npz\b",
    "numpy load": r"np\.load|numpy\.load|loadtxt|genfromtxt|fromfile",
    "loader module": r"load_model|load_weights|run_model",
    "file read": r"\bopen\s*\(|read_bytes|read_text|pickle|torch\.load",
    "dynamic import": r"importlib|__import__|exec\s*\(|eval\s*\(",
}
RUNNER = """
import sys, numpy as np
sys.path.insert(0, ".")
import twin
x = np.load(sys.argv[1])
np.save(sys.argv[2], np.asarray(twin.predict(x)).astype(np.int64).reshape(-1))
"""


def parse_estimate(text):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return None
    v = float(m.group())
    return v / 100 if v > 1 else v


def load_run_weights(run):
    with np.load(run / "weights.npz") as f:
        return {k: f[k].astype(np.float64) for k in ("w_ih", "w_hh", "w_out")}


def static_scan(run):
    hits = []
    for py in sorted(run.glob("*.py")):
        if py.name == "load_model.py":
            continue
        src = py.read_text(errors="replace")
        for label, pat in STATIC_PATTERNS.items():
            for m in re.finditer(pat, src):
                hits.append({"file": py.name, "kind": label, "match": m.group()})
    return hits


def weight_literals(src, w):
    """How many of the weights appear in the source to 3 significant decimals."""
    nums = {round(float(s), 3) for s in re.findall(r"[-+]?\d+\.\d{3,}", src)}
    vals = np.concatenate([v.ravel() for v in w.values()])
    found = sum(1 for v in vals if round(float(v), 3) in nums or round(float(-v), 3) in nums)
    return found, len(vals)


def run_twin_isolated(run, x, timeout=300):
    """Predictions of twin.py run from a scratch dir with only the agent's .py files."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for py in run.glob("*.py"):
            if py.name != "load_model.py":
                (tmp / py.name).write_bytes(py.read_bytes())
        np.save(tmp / "_x.npy", x)
        (tmp / "_runner.py").write_text(RUNNER)
        r = subprocess.run(
            [sys.executable, "-I", "_runner.py", "_x.npy", "_y.npy"],
            cwd=tmp, capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return None, r.stderr.strip().splitlines()[-1:] or ["failed"]
        return np.load(tmp / "_y.npy"), None


def score_run(run, n=1_000_000, seed=SEED):
    run = Path(run).expanduser().resolve()
    manifest = json.loads((run.parent / f"{run.name}.manifest.json").read_text())
    w = load_run_weights(run)
    out = {
        "run": run.name,
        "n_inputs": n,
        "seed": seed,
        "weights_intact": sha256(run / "weights.npz") == manifest["weights_sha256"]
        and sha256(run / "load_model.py") == manifest["loader_sha256"],
        "explanation_present": (run / "EXPLANATION.md").is_file()
        and bool((run / "EXPLANATION.md").read_text().strip()),
    }

    # Twin vs the model. Inputs float32 like AlgZoo; both sides see the same x.
    x = np.random.default_rng(seed).standard_normal((n, manifest["seq_len"])).astype(np.float32)
    model_pred = forward(w, x.astype(np.float64)).argmax(-1)
    twin = {"ok": False, "agreement": None, "accuracy": None, "error": None}
    hits, literals, runs_alone = static_scan(run), None, None
    if (run / "twin.py").is_file():
        found, total = weight_literals((run / "twin.py").read_text(errors="replace"), w)
        literals = {"found": found, "total": total}
        try:
            pred, err = run_twin_isolated(run, x)
        except subprocess.TimeoutExpired:
            pred, err = None, ["timeout"]
        runs_alone = pred is not None
        if pred is not None and pred.shape == (n,):
            target = np.argsort(x, axis=-1)[:, manifest["seq_len"] // 2]
            twin.update(ok=True, agreement=float((pred == model_pred).mean()),
                        accuracy=float((pred == target).mean()))
        else:
            twin["error"] = err or [f"bad output shape {None if pred is None else pred.shape}"]
    else:
        twin["error"] = ["twin.py missing"]
    out["twin"] = twin
    # Flag: reads the weights (static), or cannot run without them, or pastes most of them in.
    out["cheating"] = {
        "static_hits": hits,
        "weight_literals": literals,
        "runs_without_weights": runs_alone,
        "flagged": bool(hits) or (runs_alone is False and twin["error"] != ["timeout"] and _needs_weights(twin))
        or bool(literals and literals["found"] >= literals["total"] // 2),
    }

    exact = manifest["exact_accuracy"]
    est_file = run / "estimate.txt"
    value = parse_estimate(est_file.read_text()) if est_file.is_file() else None
    out["estimate"] = {
        "value": value,
        "exact_accuracy": exact,
        "abs_error_points": None if value is None else 100 * abs(value - exact),
    }

    log = run / "model_calls.jsonl"
    calls = [json.loads(l) for l in log.read_text().splitlines() if l.strip()] if log.is_file() else []
    out["model_calls"] = {"calls": len(calls), "inputs": int(sum(c.get("n_inputs", 0) for c in calls))}
    return out


def _needs_weights(twin):
    """A twin that crashes from the scratch dir is only suspicious if the crash
    mentions the weights or the loader; other crashes are plain bugs."""
    msg = " ".join(twin["error"] or []).lower()
    return any(s in msg for s in ("weights", "npz", "load_model", "no such file", "modulenotfound"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Score an agent run directory (JSON to stdout).")
    p.add_argument("run", type=Path)
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(argv)
    print(json.dumps(score_run(a.run, n=a.n, seed=a.seed), indent=2))


if __name__ == "__main__":
    main()
