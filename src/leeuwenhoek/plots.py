"""Figures for the marimo notebooks: what an argmedian L=3 RNN does with one input.

Kept out of the notebook so they can be rendered to PNG and checked by eye
(scripts/render_plots.py). Colours follow xabier.me's ImHex theme.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from leeuwenhoek.forward import forward
from leeuwenhoek.metrics import argmedian_target

BG, PANEL, TEXT, MUTED, GRID = "#0f0f0f", "#101010", "#eaeaea", "#7f7f7f", "#3a3a3f"
COLORS = ["#4296f9", "#e0a070", "#4dc69b"]  # x1, x2, x3
ERROR, LAMP = "#e74c3c", "#f0be50"
SUB = "₁₂₃"


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_numbers(ax, x, pred):
    """The three numbers in reading order; dashed line = median; arrow = the net's pick."""
    _style(ax)
    med = int(argmedian_target(np.asarray(x)[None])[0])
    ok = pred == med
    for i, v in enumerate(x):
        ax.plot([i, i], [0, v], color=COLORS[i], lw=4, solid_capstyle="butt")
        ax.scatter([i], [v], s=420, color=COLORS[i], zorder=3)
        ax.text(i, v, f"x{SUB[i]}", ha="center", va="center", fontsize=11, fontweight="bold", color=BG, zorder=4)
    ax.axhline(0, color=GRID, lw=1)
    ax.axhline(x[med], color=MUTED, lw=1, ls=(0, (3, 4)))
    ax.text(2.45, x[med], " mediana", va="center", ha="left", color=MUTED, fontsize=10)
    lo = min(min(x), 0) - 0.9
    c = "#4dc69b" if ok else ERROR
    ax.annotate("la red", xy=(pred, lo + 0.45), xytext=(pred, lo), ha="center", va="top", color=c, fontsize=11,
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax.set_xlim(-0.6, 3.1)
    ax.set_ylim(lo - 0.5, max(max(x), 0) + 0.6)
    ax.set_xticks([])
    ax.set_title("✓ acierta" if ok else "✗ falla", color=c, fontsize=15, fontweight="bold", loc="left")


def plot_steps(ax, x, states):
    """Lamps: two neurons that fire when the value goes up, two when it goes down, per step."""
    _style(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    peak = max(states.max(), 1e-9)
    for t in range(3):
        cx = 2.0 + 2.2 * t
        ax.text(cx, 3.55, f"lee x{SUB[t]}", ha="center", color=COLORS[t], fontsize=13)
        for row, (a, b) in enumerate([(0, 1), (2, 3)]):
            cy = 2.6 - 1.2 * row
            for k, n in enumerate((a, b)):
                v = states[t, n]
                face = LAMP if v > 0 else "#161616"
                alpha = 0.35 + 0.65 * v / peak if v > 0 else 1
                ax.add_patch(plt.Circle((cx - 0.4 + 0.8 * k, cy), 0.33, facecolor=face, alpha=alpha,
                                        edgecolor=LAMP if v > 0 else GRID, lw=2))
        before = 0.0 if t == 0 else x[t - 1]
        prev = "0" if t == 0 else f"x{SUB[t - 1]}"
        ax.text(cx, 0.75, f"x{SUB[t]} {'>' if x[t] > before else '<'} {prev}", ha="center", color=MUTED, fontsize=12)
        up, down = states[t, :2].max() > 0, states[t, 2:].max() > 0
        read = "duda" if up and down else "nada" if not (up or down) else "↑" if up else "↓"
        ax.text(cx, 0.25, f"la red: {read}", ha="center", fontsize=12,
                color=ERROR if read in ("duda", "nada") else TEXT)
    ax.text(0.05, 2.6, "↑ sube", va="center", color=MUTED, fontsize=12)
    ax.text(0.05, 1.4, "↓ baja", va="center", color=MUTED, fontsize=12)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")


def answer_grid(w, x1, center, half, res=300):
    """The net's answer (0, 1, 2) and the true median over an x2-x3 square around `center`."""
    g = np.linspace(-half, half, res)
    x2, x3 = np.meshgrid(center[0] + g, center[1] - g)  # row 0 is the top
    xs = np.stack([np.full(x2.size, x1), x2.ravel(), x3.ravel()], axis=1)
    pred = forward(w, xs).argmax(-1).reshape(res, res)
    true = argmedian_target(xs).reshape(res, res)
    return pred, true


def plot_plane(ax, w, x, center, half, title):
    """Colour = number the net picks for each (x2, x3) with this x1; red = wrong."""
    _style(ax)
    pred, true = answer_grid(w, x[0], center, half)
    img = np.where(pred == true, pred, 3)
    cmap = ListedColormap([*[_dim(c) for c in COLORS], ERROR])
    ext = [center[0] - half, center[0] + half, center[1] - half, center[1] + half]
    ax.imshow(img, cmap=cmap, vmin=0, vmax=3, extent=ext, interpolation="nearest")
    ax.scatter([x[1]], [x[2]], s=160, facecolors="none", edgecolors="white", linewidths=2.5, zorder=3)
    ax.set_xlabel("x₂", color=MUTED)
    ax.set_ylabel("x₃", color=MUTED)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left")
    ax.set_xlim(ext[:2])
    ax.set_ylim(ext[2:])


def _dim(hex_color, k=0.5):
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    bg = np.array([16, 16, 16]) / 255
    return tuple(bg + (rgb - bg) * k)


def overview(w, x, zoom=0.1):
    """The whole picture for one input: numbers, lamps, answer plane and a magnified patch."""
    x = np.asarray(x, np.float32)
    logits, states = forward(w, x[None], return_states=True)
    pred = int(logits.argmax(-1)[0])
    fig = plt.figure(figsize=(13, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1.15], hspace=0.28, wspace=0.25)
    plot_numbers(fig.add_subplot(gs[0, 0]), x, pred)
    plot_steps(fig.add_subplot(gs[0, 1:]), x, states[0])
    ax = fig.add_subplot(gs[1, 0:2])
    plot_plane(ax, w, x, (0.0, 0.0), 2.5, f"qué elige la red con x₁ = {x[0]:.2f}  (rojo = error)")
    ax.add_patch(plt.Rectangle((x[1] - zoom, x[2] - zoom), 2 * zoom, 2 * zoom, fill=False, ec="white", ls="--", lw=1.2))
    plot_plane(fig.add_subplot(gs[1, 2]), w, x, (x[1], x[2]), zoom, "lupa ×25")
    for i, c in enumerate(COLORS):
        fig.text(0.13 + 0.12 * i, 0.035, f"■ elige x{SUB[i]}", color=c, fontsize=11)
    fig.text(0.49, 0.035, "■ error", color=ERROR, fontsize=11)
    return fig
