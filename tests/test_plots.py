import matplotlib

matplotlib.use("Agg")

from leeuwenhoek.models import load_argmedian  # noqa: E402
from leeuwenhoek.plots import answer_grid, overview  # noqa: E402


def test_overview_draws_for_typical_inputs():
    _, w = load_argmedian(4, 3)
    for x in ([0.9, -0.4, 0.5], [1.0, -0.3, -0.295], [1.0, 0.7, 0.705]):
        fig = overview(w, x)
        assert len(fig.axes) == 4


def test_answer_grid_marks_the_band_error():
    # the "band" input is a known error: the net says x1, the median is x3
    _, w = load_argmedian(4, 3)
    pred, true = answer_grid(w, 1.0, (-0.3, -0.295), 1e-6, res=1)
    assert pred[0, 0] == 0 and true[0, 0] == 2
