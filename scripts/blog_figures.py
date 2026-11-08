"""Figures for the post "Taking apart a 32-number network" (xabier.me), in its ImHex dark theme.

Usage: uv run python scripts/blog_figures.py --lang en --out figures/blog/en
Every number drawn is computed here from the weights, except the twin ladder,
whose values come from notes/argmedian_4_3.md (scripts twins_argmedian_4_3.py
and mechanisms_h4.py; recomputing 150 cones here would take minutes).
Palette x1/x2/x3 checked with the dataviz validator on #101010 (dark mode).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from leeuwenhoek.design import BINV, B, designed_weights, derived, read_off  # noqa: E402
from leeuwenhoek.exact import pieces  # noqa: E402
from leeuwenhoek.forward import forward  # noqa: E402
from leeuwenhoek.metrics import argmedian_target  # noqa: E402
from leeuwenhoek.models import load_argmedian  # noqa: E402
from leeuwenhoek.plots import answer_grid  # noqa: E402
from leeuwenhoek.sphere import directions, grid  # noqa: E402

BG, TEXT, TEXT2, MUTED, GRID = "#0f0f0f", "#eaeaea", "#aaaaaa", "#7f7f7f", "#3a3a3f"
X = ["#3d8ef0", "#d65f5c", "#9677dc"]  # x1, x2, x3 (validated: dark, all checks pass)
WRONG = "#eaeaea"  # errors drawn in the text colour: the brightest thing on a dark page
LAMP = "#f0be50"
SUB = "₁₂₃"

T = {
    "en": {
        "answer": "answers x{}", "wrong": "wrong",
        "zoom": "zoom on one boundary (×{})",
        "wrong_share": "share of the wrong answers",
        "steps": "reads x{}", "up": "↑ up pair", "down": "↓ down pair",
        "run": "run: up, up → x₂", "turn": "turn: up, down → x₁ vs x₃", "tie": "near tie: the pair breaks",
        "says": "says", "right": "right", "wrongw": "wrong",
        "comparison": "comparison channel (sum)", "memory": "memory channel (difference)",
        "breaks": "pair breaks", "x2": "x₂ (with x₁ = 1)", "chan": "value at step 2",
        "plane": "what the net answers with x₁ = 1", "lens": "magnifier ×{}",
        "ladder_y": "agreement with the network (%)", "rule": "median\nrule", "band3": "+ tie\nband",
        "band2": "+ step-2\nslivers", "turns": "+ turns,\ndead state", "z22": "22\nzones",
        "copy": "just copying calculations", "exact": "100 %",
        "samples": "random samples of the network", "err": "estimation error (points)",
        "sampling": "sampling (1 std. error)", "ours1": "first estimate", "ours2": "final estimate",
        "ratio": "trained ÷ predicted", "pred_note": "predicted = 1",
    },
    "es": {
        "answer": "responde x{}", "wrong": "error",
        "zoom": "lupa en una frontera (×{})",
        "wrong_share": "reparto de las respuestas erróneas",
        "steps": "lee x{}", "up": "↑ pareja sube", "down": "↓ pareja baja",
        "run": "racha: sube, sube → x₂", "turn": "giro: sube, baja → x₁ contra x₃", "tie": "casi empate: la pareja se rompe",
        "says": "dice", "right": "acierta", "wrongw": "falla",
        "comparison": "canal de comparación (suma)", "memory": "canal de memoria (diferencia)",
        "breaks": "la pareja se rompe", "x2": "x₂ (con x₁ = 1)", "chan": "valor en el paso 2",
        "plane": "qué responde la red con x₁ = 1", "lens": "lupa ×{}",
        "ladder_y": "coincidencia con la red (%)", "rule": "regla de\nla mediana", "band3": "+ banda\ndel empate",
        "band2": "+ franjas\ndel paso 2", "turns": "+ giros y\nestado muerto", "z22": "22\nzonas",
        "copy": "solo copiar cálculos", "exact": "100 %",
        "samples": "pruebas de la red al azar", "err": "error de la estimación (puntos)",
        "sampling": "muestreo (1 error típico)", "ours1": "primera estimación", "ours2": "estimación final",
        "ratio": "entrenado ÷ predicho", "pred_note": "predicho = 1",
    },
}


def setup():
    plt.rcParams.update({
        "font.family": ["JetBrainsMono Nerd Font Mono", "DejaVu Sans Mono"],
        "font.size": 11, "text.color": TEXT, "axes.labelcolor": TEXT2,
        "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def dim(c, k=0.55):
    rgb = np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    bg = np.array([15, 15, 15]) / 255
    return tuple(bg + (rgb - bg) * k)


def legend_chips(fig, s, y=0.02, x0=0.08, with_wrong=True):
    items = [(dim(X[i]), s["answer"].format(SUB[i])) for i in range(3)]
    if with_wrong:
        items.append((WRONG, s["wrong"]))
    for k, (c, label) in enumerate(items):
        x = x0 + 0.2 * k
        fig.patches.append(Rectangle((x, y), 0.014, 0.03, transform=fig.transFigure, color=c))
        fig.text(x + 0.02, y + 0.015, label, va="center", color=TEXT2, fontsize=10)


def fig_sphere(w, s, figsize=(11, 4.6)):
    """Equal-area map of input directions: colour = answer, light = wrong, plus a zoom on one boundary."""
    theta, z = grid(1440, 720)
    u = directions(theta, z)
    pred = forward(w, u.reshape(-1, 3)).argmax(-1).reshape(theta.shape)
    wrong = pred != argmedian_target(u)
    img = np.where(wrong, 3, pred)
    cmap = ListedColormap([dim(c) for c in X] + [WRONG])
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.08, left=0.03, right=0.99, top=0.86, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 360)
    ax.imshow(img, origin="lower", extent=[0, 360, -1, 1], aspect="auto", cmap=cmap, vmin=0, vmax=3,
              interpolation="nearest")
    for k in range(6):
        o = np.argsort(directions(np.array(np.radians(60 * k)), np.array(0.0)))
        label = f"x{SUB[o[0]]}<x{SUB[o[1]]}<x{SUB[o[2]]}"
        for pos, ha in ([(0, "left"), (360, "right")] if k == 0 else [(60 * k, "center")]):
            ax.text(pos, 1.06, label, ha=ha, fontsize=9, color=TEXT2)
    lo, hi = 26, 34
    ax.plot([30], [-1.07], marker="^", color=TEXT, ms=7, clip_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # zoom: same map, finer pixels, on one tie boundary (theta = 30)
    th = np.radians(np.linspace(lo, hi, 400))
    zz = np.linspace(-1, 1, 500)
    TH, ZZ = np.meshgrid(th, zz)
    uz = directions(TH, ZZ)
    pz = forward(w, uz.reshape(-1, 3)).argmax(-1).reshape(TH.shape)
    iz = np.where(pz != argmedian_target(uz), 3, pz)
    az = fig.add_subplot(gs[1])
    az.imshow(iz, origin="lower", extent=[lo, hi, -1, 1], aspect="auto", cmap=cmap, vmin=0, vmax=3,
              interpolation="nearest")
    az.set_title(s["zoom"].format(int(360 / (hi - lo))), fontsize=10, color=TEXT2)
    az.set_xticks([])
    az.set_yticks([])
    for sp in az.spines.values():
        sp.set_color(TEXT)
        sp.set_linestyle("--")
        sp.set_visible(True)
    legend_chips(fig, s)
    return fig


def fig_wrong_answers(w, s):
    """Which position the net names when it is wrong, exactly (path enumeration): mostly x1."""
    err = np.zeros(3)
    for piece in pieces(w):
        if piece["pred"] != piece["order"][1]:
            err[piece["pred"]] += piece["prob"]
    share = err / err.sum()
    print("wrong answers ->", np.round(share, 4))
    fig, ax = plt.subplots(figsize=(8, 2.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.12)
    for i in range(3):
        ax.barh(2 - i, share[i], height=0.62, color=X[i])
        ax.text(share[i] + 0.012, 2 - i, f"{share[i]:.0%}", va="center", color=TEXT, fontsize=12)
        ax.text(-0.012, 2 - i, f"x{SUB[i]}", va="center", ha="right", color=TEXT2, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(s["wrong_share"], loc="left", color=TEXT2, fontsize=11)
    return fig


def fig_lamps(w, s):
    """Which neurons fire at each step, for a run, a turn and a near tie."""
    cases = [("run", [0.2, 0.6, 1.1]), ("turn", [0.9, -0.4, 0.5]), ("tie", [1.0, -0.3, -0.295])]
    fig, ax = plt.subplots(figsize=(10.4, 10.6))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    ax.set_axis_off()
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 10.6)
    ax.set_aspect("equal")
    for t in range(3):
        ax.text(3.6 + 2.0 * t, 10.15, s["steps"].format(SUB[t]), ha="center", color=X[t], fontsize=13)
    ax.text(9.6, 10.15, s["says"], ha="center", color=TEXT2, fontsize=13)
    for row, (key, x) in enumerate(cases):
        xs = np.array([x], np.float64)
        logits, st = forward(w, xs, return_states=True)
        pred, med = int(logits.argmax()), int(argmedian_target(xs)[0])
        top = 9.3 - 3.25 * row
        ax.text(0.15, top, s[key], color=TEXT, fontsize=12)
        ax.text(0.15, top - 0.95, s["up"], color=MUTED, fontsize=10, va="center")
        ax.text(0.15, top - 1.95, s["down"], color=MUTED, fontsize=10, va="center")
        peak = st[0].max()
        for t in range(3):
            cx = 3.6 + 2.0 * t
            for r, pair in enumerate([(0, 1), (3, 2)]):  # (n1, n2) up half, (n4, n3) down half
                for k, n in enumerate(pair):
                    v = st[0, t, n]
                    on = v > 0
                    ax.add_patch(Circle((cx - 0.38 + 0.76 * k, top - 0.95 - r * 1.0), 0.3,
                                        facecolor=LAMP if on else "#161616",
                                        alpha=(0.45 + 0.55 * v / peak) if on else 1,
                                        edgecolor=LAMP if on else GRID, lw=1.5))
            ok = pred == med
        ax.add_patch(Circle((9.6, top - 1.45), 0.42, color=X[pred]))
        ax.text(9.6, top - 1.45, f"x{SUB[pred]}", ha="center", va="center", color=BG, fontsize=12, fontweight="bold")
        ax.text(9.6, top - 2.25, f"{'✓ ' + s['right'] if ok else '✗ ' + s['wrongw']}", ha="center",
                color=TEXT if ok else LAMP, fontsize=11)
        vals = "   ".join(f"x{SUB[i]}={x[i]:+.3g}" for i in range(3))
        ax.text(0.15, top - 2.6, vals, color=MUTED, fontsize=9.5)
    return fig


def fig_channels(w, s):
    """Step 2 for x1 = 1 as x2 sweeps through x1: comparison vs memory, and where the pair breaks."""
    x2 = np.linspace(0.4, 1.6, 2401)
    xs = np.stack([np.ones_like(x2), x2, np.zeros_like(x2)], axis=1)
    _, st = forward(w, xs, return_states=True)
    h1 = st[:, 0]
    pre = x2[:, None] * w["w_ih"].astype(float) + h1 @ w["w_hh"].astype(float).T
    sd = pre @ B.T  # (s+, d+, s-, d-)
    comp, mem = sd[:, 0], sd[:, 1]
    act = st[:, 1] > 0
    broken = ~((act[:, :2].all(1) & ~act[:, 2:].any(1)) | (act[:, 2:].all(1) & ~act[:, :2].any(1)))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.15)
    edges = np.flatnonzero(np.diff(broken.astype(int)))
    if broken.any():
        a, b = x2[broken].min(), x2[broken].max()
        ax.axvspan(a, b, color=LAMP, alpha=0.18, lw=0)
        ax.text((a + b) / 2, 4.6, s["breaks"], ha="center", color=LAMP, fontsize=10)
    del edges
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(1, color=GRID, lw=1, ls=":")
    ax.plot(x2, comp, color=X[0], lw=2)
    ax.plot(x2, mem, color=X[1], lw=2)
    ax.text(1.58, comp[-1] - 0.6, s["comparison"], ha="right", color=TEXT, fontsize=10)
    ax.text(1.58, mem[-1] + 0.35, s["memory"], ha="right", color=TEXT, fontsize=10)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlim(0.4, 1.6)
    ax.set_xlabel(s["x2"])
    ax.set_ylabel(s["chan"])
    return fig


def fig_plane(w, s, half_zoom=0.06):
    """Answer for every (x2, x3) with x1 = 1, plus a magnifier on the tie band."""
    fig = plt.figure(figsize=(10.5, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.22, left=0.07, right=0.98, top=0.9, bottom=0.2)
    cmap = ListedColormap([dim(c) for c in X] + [WRONG])
    center = (-0.3, -0.3)
    for k, (c, half, title) in enumerate([((0.0, 0.0), 2.0, s["plane"]),
                                          (center, half_zoom, s["lens"].format(int(2.0 / half_zoom)))]):
        ax = fig.add_subplot(gs[k])
        pred, true = answer_grid(w, 1.0, c, half, res=500)
        ax.imshow(np.where(pred == true, pred, 3), cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
                  extent=[c[0] - half, c[0] + half, c[1] - half, c[1] + half])
        ax.set_xlabel("x₂")
        ax.set_ylabel("x₃")
        ax.set_title(title, loc="left", color=TEXT2, fontsize=11)
        if k == 0:
            ax.add_patch(Rectangle((center[0] - half_zoom, center[1] - half_zoom), 2 * half_zoom, 2 * half_zoom,
                                   fill=False, ec=TEXT, lw=1.2))
    legend_chips(fig, s)
    return fig


def fig_ladder(s):
    """Agreement of each twin with the network (values from notes/argmedian_4_3.md)."""
    labels = [s["rule"], s["band3"], s["band2"], s["turns"], s["z22"], "32", "52", "136", "150"]
    vals = [98.46, 99.23, 99.33, 99.62, 99.83, 99.93, 99.97, 99.992, 100.0]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.2)
    xs = np.arange(len(vals))
    ax.axvspan(4.5, 8.5, color=GRID, alpha=0.35, lw=0)
    ax.text(6.5, 98.55, s["copy"], ha="center", color=MUTED, fontsize=10)
    ax.plot(xs[:5], vals[:5], color=X[0], lw=2, marker="o", ms=8, mec=BG, mew=2)
    ax.plot(xs[4:], vals[4:], color=MUTED, lw=2, marker="o", ms=8, mec=BG, mew=2, ls="--")
    for i, v in enumerate(vals):
        if i in (0, 4, 8):
            ax.text(i, v + 0.09, f"{v:g} %", ha="center", color=TEXT, fontsize=10)
    ax.set_xticks(xs, labels, fontsize=9)
    ax.set_ylim(98.3, 100.12)
    ax.set_yticks([98.5, 99.0, 99.5, 100.0])
    ax.set_ylabel(s["ladder_y"])
    ax.grid(axis="y", color=GRID, lw=0.6)
    return fig


def fig_estimate(s, p_err=1 - 0.98455038):
    """Sampling error vs number of samples, against estimates v1 and v2 (0.52 and 0.155 points; notes)."""
    n = np.logspace(1.5, 5, 300)
    se = np.sqrt(p_err * (1 - p_err) / n) * 100
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.15)
    ax.plot(n, se, color=X[0], lw=2)
    ax.text(n[25], se[25] * 1.08, s["sampling"], color=TEXT, fontsize=10)
    for err, key in [(0.52, "ours1"), (0.155, "ours2")]:
        n_eq = p_err * (1 - p_err) / (err / 100) ** 2
        ax.axhline(err, color=X[2], lw=1.2, ls="--")
        ax.plot([n_eq, n_eq], [0.01, err], color=MUTED, lw=1, ls=":")
        ax.plot([n_eq], [err], "o", color=X[2], ms=8, mec=BG, mew=2)
        ax.text(n[-1], err * 1.08, f"{s[key]}: {err} ", ha="right", color=TEXT, fontsize=10)
        ax.text(n_eq * 1.08, 0.035, f"≈{round(n_eq, -2):,.0f}", color=TEXT2, fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 3)
    ax.set_xlabel(s["samples"])
    ax.set_ylabel(s["err"])
    ax.grid(color=GRID, lw=0.5, which="major")
    return fig


def fig_weights(w, s):
    """Constants the design predicts from b and e, against the trained net's (one dot per place it appears)."""
    ro = read_off(w)
    k = derived(ro["a"], ro["b"], ro["e"])
    W = B @ w["w_hh"].astype(float) @ BINV
    r = w["w_out"].astype(float) @ BINV
    r = r - r[0]
    dw = designed_weights(**{n: ro[n] for n in ("a", "b", "e", "B2", "B3")}, dtype=np.float64)
    rd = dw["w_out"] @ BINV
    rd = rd - rd[0]
    rows = [
        ("p", [-W[0, 0], W[0, 2], W[2, 0], -W[2, 2]], k["p"]),
        ("q", [-W[0, 1], W[0, 3], W[2, 1], -W[2, 3]], k["q"]),
        ("c", [W[1, 0], -W[1, 2], -W[3, 0], W[3, 2]], k["c"]),
        ("α x₂", [r[1, 0], r[1, 2]], rd[1, 0]),
        ("α x₃", [r[2, 0], r[2, 2]], rd[2, 0]),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.17)
    ax.axvline(1, color=TEXT2, lw=1.2)
    ax.axvspan(0.95, 1.05, color=GRID, alpha=0.35, lw=0)
    for i, (name, trained, pred) in enumerate(rows):
        y = len(rows) - 1 - i
        ratios = np.array(trained) / pred
        ax.plot([ratios.min(), ratios.max()], [y, y], color=GRID, lw=2)
        ax.plot(ratios, [y] * len(ratios), "o", color=X[0], ms=8, mec=BG, mew=2)
        ax.text(0.78, y, name, va="center", ha="right", color=TEXT, fontsize=11)
        ax.text(1.27, y, f"{pred:+.3f}", va="center", ha="right", color=MUTED, fontsize=9)
    ax.set_xlim(0.8, 1.28)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(s["ratio"])
    ax.text(1.0, len(rows) - 0.4, s["pred_note"], ha="center", color=TEXT2, fontsize=9)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=list(T), default="en")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = T[args.lang]
    out = Path(args.out or f"figures/blog/{args.lang}")
    out.mkdir(parents=True, exist_ok=True)
    setup()
    _, w = load_argmedian(4, 3)
    w = {k: v.astype(np.float64) for k, v in w.items()}
    figs = {
        "1-sphere": fig_sphere(w, s), "2-wrong-answers": fig_wrong_answers(w, s), "3-lamps": fig_lamps(w, s),
        "4-channels": fig_channels(w, s), "5-plane": fig_plane(w, s), "6-ladder": fig_ladder(s),
        "7-estimate": fig_estimate(s), "8-weights": fig_weights(w, s),
    }
    cover = fig_sphere(w, s, figsize=(12.8, 8))  # 1280x800 cover for the projects list
    cover.savefig(out / "0-cover.png", dpi=100)
    plt.close(cover)
    for name, fig in figs.items():
        fig.savefig(out / f"{name}.png", dpi=170)
        plt.close(fig)
        print(f"saved {out / name}.png")


if __name__ == "__main__":
    main()
