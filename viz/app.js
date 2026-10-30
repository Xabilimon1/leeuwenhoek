// Page behaviour. Numbers printed as facts come from data.js (scripts/build_viz.py);
// this file only runs the forward pass (model.js) for the input you choose.
(function () {
  const D = window.LEEUWENHOEK, M = window.Model;
  const $ = (id) => document.getElementById(id);
  const sub = "₁₂₃";
  const pct = (v, d = 3) => (100 * v).toFixed(d).replace(".", ",") + " %";
  const num = (v, d = 3) => v.toFixed(d).replace(".", ",");
  const short = (v) => (Math.abs(v) < 5e-4 ? "0" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
  const P = D.design;

  const state = { net: "trained", x: [0.9, -0.4, 0.5], b: P.b, e: P.e };
  const weights = () => (state.net === "trained" ? D.trained : M.designed(P.a, state.b, state.e, P.B2, P.B3));

  const NETS = [
    { id: "trained", t: "Red entrenada", d: `zoo_argmedian(4, 3). Acierto exacto ${pct(D.exact.model_accuracy)}` },
    { id: "design", t: "Gemelo diseñado", d: `Desde la explicación, 2 números libres. Exacto ${pct(D.exact.design_accuracy)}, coincide con la red en ${pct(D.exact.design_agreement_1e6, 2)}` },
  ];
  const EXAMPLES = [
    { id: "turn", t: "Giro: x₂ es el mínimo", d: "baja y sube: la red compara x₁ con x₃", x: [0.9, -0.4, 0.5] },
    { id: "run", t: "Racha que sube", d: "sube y sube: la mediana es x₂ seguro", x: [-0.6, 0.1, 0.9] },
    { id: "band", t: "Casi empate x₂ ≈ x₃, banda", d: "se encienden las dos mitades y gana x₁", x: [1, -0.3, -0.295] },
    { id: "dead", t: "Casi empate x₂ ≈ x₃, estado muerto", d: "todo a 0 en el paso 3: gana x₁ por desempate", x: [1, 0.7, 0.705] },
    { id: "step2", t: "Casi empate x₁ ≈ x₂", d: "frontera desplazada: responde x₂ y la mediana es x₁", x: [1, 0.99, 1.5] },
  ];

  function radios(el, items, name, checked, onPick) {
    for (const it of items) {
      const label = document.createElement("label");
      label.className = "opt";
      label.innerHTML = `<input type="radio" name="${name}" value="${it.id}"><span><span class="opt-t">${it.t}</span><span class="opt-d">${it.d}</span></span>`;
      const input = label.querySelector("input");
      input.checked = it.id === checked;
      input.addEventListener("change", () => onPick(it));
      el.appendChild(label);
    }
  }

  // ---------- inside the network ----------
  function renderInside() {
    const w = weights(), f = M.forward(w, state.x);
    let hx = `<span class="h">paso</span>` + ["n₁", "n₂", "n₃", "n₄"].map((n) => `<span class="h">${n}</span>`).join("") +
      `<span class="h gap"></span>` + ["s+", "d+", "s−", "d−"].map((n) => `<span class="h">${n}</span>`).join("");
    f.hs.forEach((h, t) => {
      const sd = M.mv(M.BM, h);
      hx += `<span class="off">t=${t + 1} x${sub[t]}</span>` +
        h.map((v, i) => `<span class="${v > 0 ? "on" : "offc"}" title="preactivación ${short(f.pre[t][i])}">${short(v)}</span>`).join("") +
        `<span class="gap"></span>` + sd.map((v, i) => `<span class="${i < 2 ? "pos" : "neg"}">${short(v)}</span>`).join("");
    });
    $("hx").innerHTML = hx;

    // activation path and its linear map
    const masks = f.pre.map((z) => z.map((v) => (v > 0 ? 1 : 0)));
    const halves = masks.map((m) => (m[0] || m[1] ? 1 : 0) + (m[2] || m[3] ? 2 : 0)); // 1 = +, 2 = -, 3 = both, 0 = none
    const glyph = { 0: '<span class="z">0</span>', 1: '<span class="p">+</span>', 2: '<span class="n">−</span>', 3: '<span class="m">±</span>' };
    let kind;
    if (halves[2] === 0) kind = '<span class="err">estado muerto</span>: todo apagado en el paso 3, todos los logits valen 0 y gana x₁ por desempate';
    else if (halves.includes(3)) kind = `<span class="err">camino mezclado en el paso ${halves.indexOf(3) + 1}</span>: casi empate, las dos mitades encendidas a la vez`;
    else if (halves[1] === halves[2]) kind = '<span class="ok">racha</span>: dos veces en el mismo sentido, la mediana es x₂';
    else kind = '<span class="ok">giro</span>: x₂ es el mínimo o el máximo, decide x₁ contra x₃';
    const Mx = M.pathMatrix(w, masks);
    const term = (c, j) => `${c < 0 ? "−" : "+"}${Math.abs(c).toFixed(2)}·x${sub[j]}`;
    const lines = Mx.map((r, k) => `L${sub[k]} = ${r.map(term).join(" ").replace(/^\+/, " ")}`).join("\n");
    $("path").innerHTML =
      `<div class="pathline">${halves.map((h) => glyph[h]).join(" ")} <span class="dim" style="font-size:12px">${masks.map((m) => m.join("")).join("-")}</span></div>` +
      `<p style="font-size:13px">${kind}</p><div class="mat">${lines}</div>` +
      `<p class="note">// Dentro de un camino la red es lineal: estos tres números por logit son todo lo que hace con esta entrada y con todas las de su zona.</p>`;

    // logits
    const pred = M.argmax(f.logits), med = M.median(state.x), lim = Math.max(1e-9, ...f.logits.map(Math.abs));
    $("logits").innerHTML = f.logits.map((l, k) => {
      const wd = (50 * Math.abs(l)) / lim, left = l >= 0 ? 50 : 50 - wd;
      return `<div class="lrow"><span style="color:var(--x${k + 1})">x${sub[k]}</span><div class="track"><div class="zero"></div>` +
        `<div class="bar" style="left:${left}%;width:${wd}%;background:var(--x${k + 1});opacity:${k === pred ? 1 : 0.4}"></div></div><span class="v">${short(l)}</span></div>`;
    }).join("") +
      `<p class="verdict"><span class="tag ${pred === med ? "ok" : "err"}">${pred === med ? "OK" : "ERROR"}</span>responde <b style="color:var(--x${pred + 1})">x${sub[pred]}</b>, la mediana es <b style="color:var(--x${med + 1})">x${sub[med]}</b></p>`;
  }

  // ---------- sphere map ----------
  const map = $("map"), ctx = map.getContext("2d");
  let image = null;
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const rgb = (n) => { const h = css(n).replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };

  function computeMap() {
    const w = weights(), W = map.width, H = map.height, img = ctx.createImageData(W, H);
    const cols = [rgb("--x1"), rgb("--x2"), rgb("--x3")], bad = rgb("--error"), bg = rgb("--panel");
    let wrong = 0;
    for (let j = 0; j < H; j++) {
      const z = 1 - ((j + 0.5) / H) * 2;
      for (let i = 0; i < W; i++) {
        const u = M.direction(((i + 0.5) / W) * 2 * Math.PI, z), p = M.argmax(M.forward(w, u).logits), ok = p === M.median(u);
        if (!ok) wrong++;
        const c = ok ? cols[p].map((v, k) => Math.round(bg[k] + (v - bg[k]) * 0.42)) : bad, o = (j * W + i) * 4;
        img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
      }
    }
    image = img;
    $("mapErr").textContent = `// error en la rejilla ${pct(wrong / (W * H), 2)} (aproximado: no ve franjas más finas que un píxel)`;
  }

  function drawMap() {
    if (!image) computeMap();
    ctx.putImageData(image, 0, 0);
    const W = map.width, H = map.height;
    ctx.strokeStyle = css("--text-3"); ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    for (let k = 0; k < 6; k++) { const xx = ((30 + 60 * k) / 360) * W; ctx.beginPath(); ctx.moveTo(xx, 0); ctx.lineTo(xx, H); ctx.stroke(); }
    ctx.setLineDash([]); ctx.font = "600 12px JetBrains Mono, monospace"; ctx.textAlign = "center"; ctx.fillStyle = css("--text");
    for (let k = 0; k < 6; k++) {
      const u = M.direction((k * Math.PI) / 3, 0), o = [0, 1, 2].sort((a, b) => u[a] - u[b]);
      ctx.fillText(o.map((i) => "x" + (i + 1)).join("<"), k === 0 ? 34 : (k / 6) * W, 16);
    }
    const n = Math.hypot(...state.x);
    if (n > 0) {
      const u = state.x.map((v) => v / n), th = (Math.atan2(M.dot(u, M.E2), M.dot(u, M.E1)) + 2 * Math.PI) % (2 * Math.PI);
      const px = (th / (2 * Math.PI)) * W, py = ((1 - M.dot(u, M.E3)) / 2) * H;
      ctx.lineWidth = 4; ctx.strokeStyle = "#000"; ctx.beginPath(); ctx.arc(px, py, 9, 0, 7); ctx.stroke();
      ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.beginPath(); ctx.arc(px, py, 9, 0, 7); ctx.stroke();
    }
  }

  map.addEventListener("click", (ev) => {
    const r = map.getBoundingClientRect(), th = ((ev.clientX - r.left) / r.width) * 2 * Math.PI, z = 1 - ((ev.clientY - r.top) / r.height) * 2;
    const u = M.direction(th, Math.max(-0.999, Math.min(0.999, z))), s = 1.8 / Math.max(...u.map(Math.abs));
    setX(u.map((v) => +(v * s).toFixed(3)));
  });

  // ---------- error law ----------
  function drawLaw() {
    const L = 64, R = 660, T = 14, B = 250, x0 = 0.002, x1 = 0.2, y0 = 5e-6, y1 = 0.1;
    const lx = (v) => L + ((Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0))) * (R - L);
    const ly = (v) => B - ((Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * (B - T);
    const law = (r) => D.law.K * Math.pow(r, D.law.gamma);
    let s = "";
    for (const t of [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]) s += `<line x1="${L}" x2="${R}" y1="${ly(t)}" y2="${ly(t)}" stroke="var(--border)"/><text x="${L - 8}" y="${ly(t) + 4}" text-anchor="end">${pct(t, t < 1e-3 ? 3 : 1)}</text>`;
    for (const t of [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]) s += `<line x1="${lx(t)}" x2="${lx(t)}" y1="${T}" y2="${B}" stroke="var(--grid-major)"/><text x="${lx(t)}" y="${B + 16}" text-anchor="middle">${String(t).replace(".", ",")}</text>`;
    s += `<text x="${(L + R) / 2}" y="${B + 36}" text-anchor="middle">b / a: tamaño de la memoria frente a la comparación</text>`;
    s += `<line x1="${lx(x0)}" y1="${ly(law(x0))}" x2="${lx(x1)}" y2="${ly(Math.min(y1, law(x1)))}" stroke="var(--accent)" stroke-width="2"/>`;
    for (const [r, e] of D.law.points) s += `<rect x="${lx(r) - 4}" y="${ly(e) - 4}" width="8" height="8" fill="var(--known)"/>`;
    const tr = P.b / P.a, te = 1 - D.exact.model_accuracy;
    s += `<rect x="${lx(tr) - 6}" y="${ly(te) - 6}" width="12" height="12" transform="rotate(45 ${lx(tr)} ${ly(te)})" fill="var(--error)"/>` +
      `<text x="${lx(tr) - 12}" y="${ly(te) + 4}" text-anchor="end" style="fill:var(--error)">red entrenada ${pct(te, 2)}</text>`;
    if (state.net === "design") {
      const r = state.b / P.a;
      if (r >= x0 && r <= x1) s += `<circle cx="${lx(r)}" cy="${ly(law(r))}" r="8" fill="none" stroke="var(--char)" stroke-width="2.5"/><text x="${lx(r) + 12}" y="${ly(law(r)) + 16}" style="fill:var(--char)">tu b: ley ${pct(law(r), 3)}</text>`;
    }
    $("law").innerHTML = s;
    $("lawNote").textContent = `// Cuadrados: error exacto de redes diseñadas con e = ${num(P.e)}. Recta: error = ${num(D.law.K)}·(b/a)^${num(D.law.gamma)}. ` +
      `En la red entrenada la ley predice ${pct(law(tr), 3)}; tiene ${pct(te, 3)}. Con b pequeño la misma red de 32 parámetros casi no falla.`;
  }

  // ---------- controls ----------
  function setX(x, fromExample) {
    state.x = x;
    ["x1", "x2", "x3"].forEach((id, i) => { $(id).value = x[i]; $(id + "n").value = x[i]; });
    if (!fromExample) document.querySelectorAll('input[name="example"]').forEach((r) => (r.checked = false));
    renderInside(); drawMap();
  }
  let pending = false;
  function refreshNet() {
    image = null;
    const k = M.designed(P.a, state.b, state.e, P.B2, P.B3).derived;
    $("derived").innerHTML = [["c", k.c], ["g", k.g], ["h = b + g", k.h], ["p", k.p], ["q", k.q]].map(([n, v]) => `<dt>${n}</dt><dd>${num(v)}</dd>`).join("");
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; renderInside(); computeMap(); drawMap(); drawLaw(); });
  }

  radios($("nets"), NETS, "net", state.net, (it) => {
    state.net = it.id;
    $("designPanel").setAttribute("aria-disabled", it.id !== "design");
    ["b", "bn", "e", "en"].forEach((id) => ($(id).disabled = it.id !== "design"));
    refreshNet();
  });
  radios($("examples"), EXAMPLES, "example", "turn", (it) => setX(it.x, true));
  ["x1", "x2", "x3"].forEach((id, i) => {
    $(id).addEventListener("input", () => { const x = [...state.x]; x[i] = +$(id).value; setX(x); });
    $(id + "n").addEventListener("change", () => { const x = [...state.x]; x[i] = Math.max(-2, Math.min(2, +$(id + "n").value || 0)); setX(x); });
  });
  for (const k of ["b", "e"]) {
    $(k).value = state[k]; $(k + "n").value = +state[k].toFixed(3);
    $(k).addEventListener("input", () => { state[k] = +$(k).value; $(k + "n").value = state[k]; refreshNet(); });
    $(k + "n").addEventListener("change", () => { state[k] = +$(k + "n").value; $(k).value = state[k]; refreshNet(); });
    $(k).disabled = $(k + "n").disabled = true;
  }
  $("readoff").textContent = `Leído de la red: a = ${num(P.a)}, b = ${num(P.b)}, e = ${num(P.e)} (mismo lado ${num(P.e_same)}, lado cruzado ${num(P.e_cross)}). ` +
    `Baja b y mira cómo se estrechan las líneas rojas del mapa.`;
  $("foot").textContent = `// Datos: viz/data.js, generado por scripts/build_viz.py el ${D.generated} (generado sobre el commit ${D.commit}). ` +
    `Estimación por los ${D.exact.mechanism_cones} conos de los mecanismos: ${pct(D.exact.mechanism_estimate)} frente al acierto exacto ${pct(D.exact.model_accuracy)}. ` +
    `Tras cada cambio: uv run python scripts/build_viz.py y recarga.`;

  refreshNet();
  setX(state.x, true);
})();
