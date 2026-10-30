import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from leeuwenhoek.design import designed_weights, read_off
    from leeuwenhoek.exact import exact_accuracy
    from leeuwenhoek.models import load_argmedian
    from leeuwenhoek.plots import overview

    _, trained = load_argmedian(4, 3)
    p = read_off(trained)
    return designed_weights, exact_accuracy, mo, overview, p, plt, trained


@app.cell
def _(mo, p):
    net = mo.ui.radio(["red entrenada", "red diseñada"], value="red entrenada", inline=True)
    x1 = mo.ui.slider(-2, 2, step=0.005, value=0.9, label="x₁", show_value=True, full_width=True)
    x2 = mo.ui.slider(-2, 2, step=0.005, value=-0.4, label="x₂", show_value=True, full_width=True)
    x3 = mo.ui.slider(-2, 2, step=0.005, value=0.5, label="x₃", show_value=True, full_width=True)
    memory = mo.ui.slider(0.02, 1.5, step=0.01, value=round(p["b"], 2), label="memoria (solo red diseñada)",
                          show_value=True, full_width=True)
    mo.hstack([mo.vstack([net, memory]), mo.vstack([x1, x2, x3])], widths=[1, 2], gap=2)
    return memory, net, x1, x2, x3


@app.cell
def _(designed_weights, exact_accuracy, memory, mo, net, overview, p, plt, trained, x1, x2, x3):
    w = trained if net.value == "red entrenada" else designed_weights(p["a"], memory.value, p["e"], p["B2"], p["B3"])
    acc, _, _ = exact_accuracy(w)
    fig = overview(w, [x1.value, x2.value, x3.value])
    plt.close(fig)
    mo.vstack([mo.md(f"**{net.value}**: acierta el **{100 * acc:.2f} %** de las veces (exacto)"), fig])
    return


if __name__ == "__main__":
    app.run()
