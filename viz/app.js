// Page behaviour. Facts (accuracy, read-off constants, the error law) come from
// data.js (scripts/build_viz.py); the page only runs the network (model.js).
(function () {
  const D = window.LEEUWENHOEK, M = window.Model, P = D.design;
  const $ = (id) => document.getElementById(id);
  const SUB = "₁₂₃", COL = ["var(--x1)", "var(--x2)", "var(--x3)"];
  const fmt = (v) => (v >= 0 ? " " : "−") + Math.abs(v).toFixed(2).replace(".", ",");
  const pct = (v, d) => (100 * v).toFixed(d).replace(".", ",") + " %";
  const svgNS = "http://www.w3.org/2000/svg";

  const state = { net: "trained", b: P.b, x: [0.9, -0.4, 0.5] };
  const weights = () => (state.net === "trained" ? D.trained : M.designed(P.a, state.b, P.e, P.B2, P.B3));

  // ---------- three numbers: a lollipop per number, in reading order ----------
  const LX = [85, 175, 265], VMAX = 2.5, Y0 = 60, Y1 = 360;
  const yOf = (v) => Y0 + ((VMAX - v) / (2 * VMAX)) * (Y1 - Y0);
  const vOf = (y) => Math.max(-VMAX, Math.min(VMAX, VMAX - ((y - Y0) / (Y1 - Y0)) * 2 * VMAX));

  function drawLine(pred, med) {
    const ok = pred === med;
    let s = `<text x="170" y="26" text-anchor="middle" font-size="18" font-weight="700" fill="${ok ? "var(--known)" : "var(--error)"}">${ok ? "✓ acierta" : "✗ falla"}</text>`;
    s += `<line x1="40" x2="310" y1="${yOf(0)}" y2="${yOf(0)}" stroke="var(--border)"/>`;
    s += `<text x="34" y="${yOf(0) + 4}" text-anchor="end" font-size="11" fill="var(--text-3)">0</text>`;
    const mv = state.x[med];
    s += `<line x1="40" x2="310" y1="${yOf(mv)}" y2="${yOf(mv)}" stroke="var(--text-3)" stroke-dasharray="3 4"/>`;
    s += `<text x="312" y="${yOf(mv) - 6}" text-anchor="end" font-size="11" fill="var(--text-2)">mediana</text>`;
    state.x.forEach((v, i) => {
      const y = yOf(v);
      s += `<line x1="${LX[i]}" x2="${LX[i]}" y1="${yOf(0)}" y2="${y}" stroke="${COL[i]}" stroke-width="4"/>`;
      s += `<g class="handle" data-i="${i}" style="cursor:ns-resize"><circle cx="${LX[i]}" cy="${y}" r="17" fill="${COL[i]}"/>` +
        `<circle cx="${LX[i]}" cy="${y}" r="26" fill="transparent"/>` +
        `<text x="${LX[i]}" y="${y + 5}" text-anchor="middle" font-size="14" font-weight="700" fill="#0f0f0f">x${SUB[i]}</text></g>`;
      s += `<text x="${LX[i]}" y="${v >= 0 ? y - 26 : y + 36}" text-anchor="middle" font-size="13" fill="var(--text-2)">${fmt(v)}</text>`;
    });
    // the network's pick: an arrow under its column
    s += `<text x="${LX[pred]}" y="395" text-anchor="middle" font-size="20" fill="${ok ? "var(--known)" : "var(--error)"}">▲</text>` +
      `<text x="${LX[pred]}" y="418" text-anchor="middle" font-size="13" fill="${ok ? "var(--known)" : "var(--error)"}">la red</text>`;
    $("line").innerHTML = s;
  }

  let drag = null;
  $("line").addEventListener("pointerdown", (ev) => {
    const h = ev.target.closest(".handle");
    if (!h) return;
    drag = +h.dataset.i;
    $("line").setPointerCapture(ev.pointerId);
  });
  $("line").addEventListener("pointermove", (ev) => {
    if (drag === null) return;
    const pt = new DOMPoint(ev.clientX, ev.clientY).matrixTransform($("line").getScreenCTM().inverse());
    const x = [...state.x];
    x[drag] = +vOf(pt.y).toFixed(3);
    setX(x);
  });
  $("line").addEventListener("pointerup", () => (drag = null));

  // ---------- the network reads them one by one ----------
  function drawSteps(f) {
    const CX = [165, 365, 565], peak = Math.max(1e-9, ...f.hs.flat());
    let s = `<text x="20" y="168" font-size="15" fill="var(--text-2)">↑ sube</text><text x="20" y="288" font-size="15" fill="var(--text-2)">↓ baja</text>`;
    const groups = f.hs.map((h) => [h[0] > 0 || h[1] > 0, h[2] > 0 || h[3] > 0]);
    f.hs.forEach((h, t) => {
      const cx = CX[t], prev = t === 0 ? "0" : `x${SUB[t - 1]}`;
      s += `<text x="${cx}" y="40" text-anchor="middle" font-size="16" fill="${COL[t]}">lee x${SUB[t]}</text>`;
      if (t > 0) s += `<text x="${(CX[t - 1] + cx) / 2}" y="228" text-anchor="middle" font-size="22" fill="var(--text-3)">→</text>`;
      [[0, 1, 160], [2, 3, 280]].forEach(([a, b, y]) => {
        [a, b].forEach((i, k) => {
          const on = h[i] > 0, glow = on ? 0.35 + 0.65 * (h[i] / peak) : 0;
          s += `<circle cx="${cx - 28 + 56 * k}" cy="${y}" r="22" fill="${on ? `rgba(240, 190, 80, ${glow.toFixed(2)})` : "#161616"}" stroke="${on ? "#f0be50" : "var(--border)"}" stroke-width="2"/>`;
        });
      });
      // what is actually true about the number just read
      const cur = state.x[t], before = t === 0 ? 0 : state.x[t - 1];
      s += `<text x="${cx}" y="350" text-anchor="middle" font-size="14" fill="var(--text-2)">x${SUB[t]} ${cur > before ? ">" : "<"} ${prev}</text>`;
      const [up, down] = groups[t];
      const read = up && down ? ["duda", "var(--error)"] : !up && !down ? ["nada", "var(--error)"] : [up ? "↑" : "↓", "var(--text)"];
      s += `<text x="${cx}" y="378" text-anchor="middle" font-size="14" fill="${read[1]}">la red: ${read[0]}</text>`;
    });
    // one line: what the pattern means
    const g = groups.map(([u, d]) => (u && d ? "both" : !u && !d ? "none" : u ? "up" : "down"));
    let msg, bad = false;
    if (g[2] === "none") { msg = "todo apagado al final: elige x₁ por defecto"; bad = true; }
    else if (g.includes("both")) { msg = `casi empate en el paso ${g.indexOf("both") + 1}: se encienden los dos grupos`; bad = true; }
    else if (g[1] === g[2]) msg = g[2] === "up" ? "sube y sube: el del medio es x₂" : "baja y baja: el del medio es x₂";
    else msg = g[1] === "down" ? "baja y sube: x₂ es el menor, gana el menor de x₁ y x₃" : "sube y baja: x₂ es el mayor, gana el mayor de x₁ y x₃";
    s += `<text x="320" y="425" text-anchor="middle" font-size="15" fill="${bad ? "var(--error)" : "var(--known)"}">${msg}</text>`;
    $("steps").innerHTML = s;
  }

  // ---------- planes: what the network picks for every x2, x3 ----------
  const RGB = { x: [[66, 150, 249], [224, 160, 112], [77, 198, 155]], err: [231, 76, 60], bg: [16, 16, 16] };
  const RES = 360, PLANE = 2.5, LOUPE = 0.1;
  const plane = $("plane"), loupe = $("loupe");
  let planeImg = null, planeKey = "";

  function paint(cv, cx, cy, half) {
    const w = weights(), x1 = state.x[0], img = new ImageData(RES, RES);
    for (let j = 0; j < RES; j++) {
      const x3 = cy + half - ((j + 0.5) / RES) * 2 * half;
      for (let i = 0; i < RES; i++) {
        const x2 = cx - half + ((i + 0.5) / RES) * 2 * half;
        const p = M.predict(w, x1, x2, x3), ok = p === M.median([x1, x2, x3]);
        const c = ok ? RGB.x[p].map((v, k) => Math.round(RGB.bg[k] + (v - RGB.bg[k]) * 0.5)) : RGB.err, o = (j * RES + i) * 4;
        img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
      }
    }
    return img;
  }

  function blit(cv, img) {
    const tmp = new OffscreenCanvas(RES, RES);
    tmp.getContext("2d").putImageData(img, 0, 0);
    const ctx = cv.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, cv.width, cv.height);
    return ctx;
  }

  function marker(ctx, px, py) {
    ctx.lineWidth = 5; ctx.strokeStyle = "#000"; ctx.beginPath(); ctx.arc(px, py, 12, 0, 7); ctx.stroke();
    ctx.lineWidth = 2.5; ctx.strokeStyle = "#fff"; ctx.beginPath(); ctx.arc(px, py, 12, 0, 7); ctx.stroke();
  }

  function axes(ctx, W) {
    ctx.font = "600 22px JetBrains Mono, monospace"; ctx.fillStyle = "#eaeaea"; ctx.shadowColor = "#000"; ctx.shadowBlur = 4;
    ctx.textAlign = "right"; ctx.fillText("x₂ →", W - 14, W - 14);
    ctx.textAlign = "left"; ctx.fillText("↑ x₃", 14, 30);
    ctx.shadowBlur = 0;
  }

  function drawPlane() {
    const key = `${state.net}|${state.b}|${state.x[0]}`;
    if (key !== planeKey) { planeImg = paint(plane, 0, 0, PLANE); planeKey = key; }
    const ctx = blit(plane, planeImg), W = plane.width, sc = W / (2 * PLANE);
    const px = (state.x[1] + PLANE) * sc, py = (PLANE - state.x[2]) * sc;
    ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.setLineDash([6, 5]);
    ctx.strokeRect(px - LOUPE * sc, py - LOUPE * sc, 2 * LOUPE * sc, 2 * LOUPE * sc);
    ctx.setLineDash([]);
    marker(ctx, px, py);
    axes(ctx, W);
  }

  function drawLoupe() {
    const ctx = blit(loupe, paint(loupe, state.x[1], state.x[2], LOUPE)), W = loupe.width;
    marker(ctx, W / 2, W / 2);
    axes(ctx, W);
  }

  function pick(cv, ev, cx, cy, half) {
    const r = cv.getBoundingClientRect();
    const x2 = cx - half + ((ev.clientX - r.left) / r.width) * 2 * half;
    const x3 = cy + half - ((ev.clientY - r.top) / r.height) * 2 * half;
    const clamp = (v) => +Math.max(-VMAX, Math.min(VMAX, v)).toFixed(4);
    setX([state.x[0], clamp(x2), clamp(x3)]);
  }
  for (const [cv, isPlane] of [[plane, true], [loupe, false]]) {
    let down = false, anchor = null;
    cv.addEventListener("pointerdown", (ev) => {
      down = true; cv.setPointerCapture(ev.pointerId);
      anchor = { x2: state.x[1], x3: state.x[2] };
      if (isPlane) pick(cv, ev, 0, 0, PLANE); else pick(cv, ev, anchor.x2, anchor.x3, LOUPE);
    });
    cv.addEventListener("pointermove", (ev) => {
      if (!down) return;
      if (isPlane) pick(cv, ev, 0, 0, PLANE); else pick(cv, ev, anchor.x2, anchor.x3, LOUPE);
    });
    cv.addEventListener("pointerup", () => (down = false));
  }

  // ---------- status bar and modes ----------
  function drawStatus() {
    let acc;
    if (state.net === "trained") acc = `acierta <b>${pct(D.exact.model_accuracy, 2)}</b> de las veces`;
    else acc = `acierta <b>≈ ${pct(1 - D.law.K * Math.pow(state.b / P.a, D.law.gamma), 2)}</b> de las veces, memoria <b>${state.b.toFixed(2).replace(".", ",")}</b> (la entrenada usa ${P.b.toFixed(2).replace(".", ",")})`;
    $("status").innerHTML = `<span>${acc}</span><span class="src">datos: scripts/build_viz.py, ${D.generated}, commit ${D.commit}</span>`;
  }

  let frame = 0;
  function render() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      const f = M.forward(weights(), state.x), pred = M.argmax(f.logits), med = M.median(state.x);
      drawLine(pred, med); drawSteps(f); drawPlane(); drawLoupe(); drawStatus();
    });
  }
  function setX(x) { state.x = x; render(); }

  function setNet(net) {
    state.net = net;
    $("m-trained").setAttribute("aria-pressed", net === "trained");
    $("m-design").setAttribute("aria-pressed", net === "design");
    $("memory").hidden = net !== "design";
    render();
  }
  $("m-trained").addEventListener("click", () => setNet("trained"));
  $("m-design").addEventListener("click", () => setNet("design"));
  $("b").value = P.b;
  $("b").addEventListener("input", () => { state.b = +$("b").value; render(); });

  render();
})();
