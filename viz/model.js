// The network maths the page runs, mirroring src/leeuwenhoek (forward.py,
// design.py, linear.py, sphere.py). tests/test_viz.py runs this file in node
// and checks it against the Python code, so the page cannot drift from it.
(function (g) {
  // (n1, n2, n3, n4) <-> (s+, d+, s-, d-), as design.B and design.BINV
  const BM = [[1, 1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]];
  const BI = [[0.5, -0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, -0.5]];
  const mm = (X, Y) => X.map((r) => Y[0].map((_, j) => r.reduce((s, v, k) => s + v * Y[k][j], 0)));
  const mv = (X, v) => X.map((r) => r.reduce((s, a, k) => s + a * v[k], 0));

  // design.designed_weights
  function designed(a, b, e, B2, B3) {
    const c = (b * (1 - e + e * e)) / (e * a), gg = c * a - e * b, q = a / (gg + b), p = (q * gg) / a;
    const S = [[-p, -q], [c, -e]], C = [[p, q], [-c, e]];
    const Wsd = [[...S[0], ...C[0]], [...S[1], ...C[1]], [...C[0], ...S[0]], [...C[1], ...S[1]]];
    const alpha = [0, (-B2 * b) / a, (-B3 * gg) / a], beta = [0, B2, -B3];
    const Usd = [0, 1, 2].map((k) => [alpha[k], beta[k], alpha[k], beta[k]]);
    return { w_ih: mv(BI, [a, b, -a, -b]), w_hh: mm(mm(BI, Wsd), BM), w_out: mm(Usd, BM), derived: { c, g: gg, q, p, h: b + gg } };
  }

  // forward.forward with return_states=True; pre = pre-activations
  function forward(w, x) {
    let h = new Array(w.w_ih.length).fill(0);
    const pre = [], hs = [];
    for (let t = 0; t < x.length; t++) {
      const z = w.w_hh.map((r, i) => w.w_ih[i] * x[t] + r.reduce((s, v, k) => s + v * h[k], 0));
      h = z.map((v) => Math.max(0, v));
      pre.push(z);
      hs.push(h);
    }
    return { pre, hs, logits: w.w_out.map((r) => r.reduce((s, v, k) => s + v * h[k], 0)) };
  }

  // linear.path_matrices: inside one ReLU path, logits = M x
  function pathMatrix(w, masks) {
    const n = w.w_ih.length, L = masks.length;
    let S = Array.from({ length: n }, () => new Array(L).fill(0));
    for (let t = 0; t < L; t++) {
      const Z = mm(w.w_hh, S);
      for (let i = 0; i < n; i++) Z[i][t] += w.w_ih[i];
      S = Z.map((row, i) => row.map((v) => v * masks[t][i]));
    }
    return mm(w.w_out, S);
  }

  const argmax = (v) => v.reduce((bi, x, i) => (x > v[bi] ? i : bi), 0); // ties -> first, as numpy
  const median = (x) => [0, 1, 2].sort((i, j) => x[i] - x[j])[1];

  // sphere.py: equal-area map coordinates
  const E1 = [1 / Math.SQRT2, -1 / Math.SQRT2, 0];
  const E2 = [1, 1, -2].map((v) => v / Math.sqrt(6));
  const E3 = [1, 1, 1].map((v) => v / Math.sqrt(3));
  const direction = (theta, z) => {
    const r = Math.sqrt(1 - z * z);
    return [0, 1, 2].map((i) => r * (Math.cos(theta) * E1[i] + Math.sin(theta) * E2[i]) + z * E3[i]);
  };
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

  g.Model = { BM, BI, mv, designed, forward, pathMatrix, argmax, median, direction, dot, E1, E2, E3 };
})(typeof window !== "undefined" ? window : globalThis);
