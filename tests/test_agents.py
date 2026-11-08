"""Agent-study tooling: clean run directories and objective scoring.

The run directory is what an agent sees, so these tests check that it holds
only the weights, a standalone loader and the task, and that the scorer
catches a twin that loads the weights instead of reimplementing them.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from agents.make_run import ALLOWED_FILES, build_run
from agents.score import parse_estimate, score_run
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.forward import forward
from leeuwenhoek.models import load_argmedian

N = 200_000  # enough to pin agreement to +-0.002 at 5 standard errors

CLEAN_MEDIAN_TWIN = '''
import numpy as np

def predict(x: np.ndarray) -> np.ndarray:
    return np.argsort(x, axis=-1)[:, 1]
'''

CHEATING_TWIN = '''
import numpy as np

_w = np.load("weights.npz")

def predict(x):
    h = np.zeros((x.shape[0], _w["w_hh"].shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(0, x[:, t, None] * _w["w_ih"] + h @ _w["w_hh"].T)
    return (h @ _w["w_out"].T).argmax(-1)
'''

SNEAKY_TWIN = '''
import numpy as np
import importlib

_m = importlib.import_module("load" + "_model")

def predict(x):
    return _m.predict(x)
'''


@pytest.fixture(scope="module")
def weights():
    _, w = load_argmedian(4, 3)
    return w


@pytest.fixture(scope="module")
def exact(weights):
    return exact_accuracy(weights)[0]


@pytest.fixture
def run(tmp_path, weights, exact):
    return build_run(
        tmp_path, agent="fake", model="argmedian_4_3", n=1,
        weights=weights, exact_acc=exact, git=False,
    )


def test_run_dir_holds_only_allowed_files(run):
    assert sorted(p.name for p in run.iterdir()) == sorted(ALLOWED_FILES)
    assert run.name == "fake-argmedian_4_3-1"


def test_run_dir_has_no_trace_of_our_explanation(run):
    text = " ".join((run / f).read_text() for f in ("load_model.py", "TASK.md")).lower()
    for word in ("leeuwenhoek", "notes/", "twins", "mechanism", "98.45", "cone", "mirror"):
        assert word not in text, word


def test_manifest_lives_outside_the_run_dir(run, exact):
    manifest = json.loads((run.parent / f"{run.name}.manifest.json").read_text())
    assert manifest["exact_accuracy"] == pytest.approx(exact)
    assert not any("manifest" in p.name for p in run.iterdir())


def test_standalone_loader_matches_our_forward(run, weights, tmp_path):
    # Runs isolated (-I) with only numpy: no torch, no alg_zoo, no leeuwenhoek.
    x = np.random.default_rng(0).standard_normal((1000, 3)).astype(np.float32)
    np.save(tmp_path / "x.npy", x)
    code = (
        "import sys, numpy as np; sys.path.insert(0, '.');"
        "import load_model as m;"
        f"np.save({str(tmp_path / 'y.npy')!r}, m.run_model(np.load({str(tmp_path / 'x.npy')!r})))"
    )
    subprocess.run([sys.executable, "-I", "-c", code], cwd=run, check=True)
    ours = forward({k: v.astype(np.float64) for k, v in weights.items()}, x.astype(np.float64))
    np.testing.assert_allclose(np.load(tmp_path / "y.npy"), ours, rtol=1e-6, atol=1e-6)
    log = (run / "model_calls.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["n_inputs"] == 1000


def test_clean_median_twin_agrees_like_the_model_accuracy(run, exact):
    (run / "twin.py").write_text(CLEAN_MEDIAN_TWIN)
    (run / "estimate.txt").write_text("98.0%\n")
    s = score_run(run, n=N)
    assert s["twin"]["ok"]
    assert abs(s["twin"]["agreement"] - exact) < 0.002
    assert s["cheating"]["flagged"] is False
    assert s["estimate"]["value"] == pytest.approx(0.98)
    assert s["estimate"]["abs_error_points"] == pytest.approx(100 * abs(0.98 - exact))


def test_twin_that_loads_weights_is_flagged(run):
    (run / "twin.py").write_text(CHEATING_TWIN)
    s = score_run(run, n=10_000)
    assert s["cheating"]["flagged"] is True
    assert s["cheating"]["static_hits"]
    assert s["cheating"]["runs_without_weights"] is False


def test_obfuscated_loader_is_caught_dynamically(run):
    (run / "twin.py").write_text(SNEAKY_TWIN)
    s = score_run(run, n=10_000)
    assert s["cheating"]["flagged"] is True
    assert s["cheating"]["runs_without_weights"] is False


def test_tampered_weights_are_detected(run):
    w = dict(np.load(run / "weights.npz"))
    w["w_out"] = w["w_out"] * 0
    np.savez(run / "weights.npz", **w)
    (run / "twin.py").write_text(CLEAN_MEDIAN_TWIN)
    s = score_run(run, n=10_000)
    assert s["weights_intact"] is False


def test_missing_deliverables_are_reported_not_crashed(run):
    s = score_run(run, n=10_000)
    assert s["twin"]["ok"] is False
    assert s["estimate"]["value"] is None
    assert s["explanation_present"] is False


@pytest.mark.parametrize("text, value", [
    ("0.9846", 0.9846), ("98.46%", 0.9846), ("accuracy: 98.46 %\n", 0.9846),
    ("Estimated accuracy = 0.97\nmethod: ...", 0.97), ("no number here", None),
])
def test_parse_estimate(text, value):
    assert parse_estimate(text) == (pytest.approx(value) if value is not None else None)


def test_scorer_cli_prints_json(run):
    (run / "twin.py").write_text(CLEAN_MEDIAN_TWIN)
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-m", "agents.score", str(run), "--n", "10000"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    assert json.loads(out)["twin"]["ok"] is True


def test_sampling_error_at_same_compute():
    from agents.score import sampling_error_points

    # 1 std. error of sampling, in points: 100 * sqrt(p (1 - p) / n)
    assert abs(sampling_error_points(0.98455, 6_300) - 0.155) < 0.002
    assert sampling_error_points(0.98455, 0) is None
