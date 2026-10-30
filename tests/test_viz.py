"""The local page's JavaScript (viz/model.js) must compute what the Python code computes."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from leeuwenhoek.design import designed_weights, read_off
from leeuwenhoek.forward import forward
from leeuwenhoek.linear import parse_path, path_matrices
from leeuwenhoek.models import load_argmedian

VIZ = Path(__file__).resolve().parents[1] / "viz"
XS = [[0.9, -0.4, 0.5], [1, -0.3, -0.295], [1, 0.7, 0.705], [-0.6, 0.1, 0.9], [0.2, 1.3, -0.8]]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_model_js_matches_python():
    _, w = load_argmedian(4, 3)
    p = read_off(w)
    trained = {k: v.astype(np.float64).tolist() for k, v in w.items()}
    js = (VIZ / "model.js").read_text() + f"""
const W = {json.dumps(trained)}, P = {json.dumps(p)}, XS = {json.dumps(XS)};
const D = Model.designed(P.a, P.b, P.e, P.B2, P.B3);
console.log(JSON.stringify({{
  trained: XS.map(x => Model.forward(W, x).logits),
  designed: XS.map(x => Model.forward(D, x).logits),
  w_hh: D.w_hh,
  M: Model.pathMatrix(W, [[1,1,0,0],[0,0,1,1],[1,1,0,0]]),
}}));
"""
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    xs = np.array(XS)
    w64 = {k: v.astype(np.float64) for k, v in w.items()}
    d = designed_weights(p["a"], p["b"], p["e"], p["B2"], p["B3"], dtype=np.float64)
    np.testing.assert_allclose(out["trained"], forward(w64, xs), atol=1e-9)
    np.testing.assert_allclose(out["designed"], forward(d, xs), atol=1e-9)
    np.testing.assert_allclose(out["w_hh"], d["w_hh"], atol=1e-12)
    M, _ = path_matrices(w64, parse_path("1100-0011-1100"))
    np.testing.assert_allclose(out["M"], M, atol=1e-9)
