"""Unit tests for :mod:`ruleofthumb.plot` (Agg backend, no display)."""

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ruleofthumb import fit_tabular, plot


@pytest.fixture()
def tabular_case():
    rng = np.random.RandomState(0)
    x = rng.rand(40, 4).astype(np.float32)
    y = ((x[:, 0] + x[:, 1]) > 1.0).astype(np.int64)
    explainer = fit_tabular(y, x, epochs=30, batch_size=40, learning_rate=0.05, seed=0)
    yield explainer, x
    plt.close("all")


@pytest.mark.parametrize(
    "name",
    ["waterfall", "force", "decision"],
)
def test_single_row_tabular_plots_return_figures(tabular_case, name):
    explainer, x = tabular_case
    fig = getattr(plot, name)(explainer, x[:1])
    assert isinstance(fig, Figure)


@pytest.mark.parametrize(
    "name",
    ["bar", "beeswarm"],
)
def test_batch_tabular_plots_return_figures(tabular_case, name):
    explainer, x = tabular_case
    fig = getattr(plot, name)(explainer, x[:10])
    assert isinstance(fig, Figure)


def test_values_and_base_use_class_bias(tabular_case):
    """RoT's SHAP-analogue baseline is the class bias g[k]; values are the signed importances."""
    explainer, x = tabular_case
    values, base = plot._values_and_base(explainer, x[:1], class_idx=1)
    assert base == pytest.approx(float(explainer.model.g[1].item()))
    assert np.allclose(values, explainer.get_explanation(x[:1])[0])


def test_text_html_sign_colouring():
    tokens = ["good", "bad", "meh"]
    importance = np.array([0.5, -0.5, 0.0])
    html = plot.text_html(importance, tokens)
    html = getattr(html, "data", html)  # IPython.display.HTML aware

    assert "good" in html and "bad" in html and "meh" in html
    good_style = _span_style(html, "good")
    bad_style = _span_style(html, "bad")
    meh_style = _span_style(html, "meh")
    assert good_style != bad_style  # opposite signs must be coloured differently
    assert good_style != meh_style  # nonzero differs from zero


def _span_style(html, token):
    marker = f">{token}<"
    position = html.index(marker)
    start = html.rindex("<span", 0, position)
    return html[start:position]


def test_text_html_max_tokens_truncates():
    tokens = [f"w{i}" for i in range(10)]
    importance = np.linspace(-1, 1, 10)
    html = getattr(plot.text_html(importance, tokens, max_tokens=4), "data", "")
    shown = sum(1 for token in tokens if f">{token}<" in html)
    assert shown == 4


def test_text_matplotlib_returns_figure():
    tokens = ["good", "bad"]
    importance = np.array([0.5, -0.5])
    fig = plot.text_matplotlib(importance, tokens)
    assert isinstance(fig, Figure)


def test_saliency_returns_figure_with_and_without_image():
    rng = np.random.RandomState(0)
    heat = rng.randn(8, 8).astype(np.float32)
    image = rng.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)

    assert isinstance(plot.saliency(heat), Figure)
    assert isinstance(plot.saliency(heat, image=image), Figure)


def test_saliency_rejects_nonpositive_power():
    with pytest.raises(ValueError, match="power"):
        plot.saliency(np.ones((4, 4)), power=0.0)


def test_word_clouds_returns_figure():
    importance_rows = [np.array([0.5, -0.2]), np.array([0.1, 0.3])]
    tokens_lists = [["good", "bad"], ["good", "great"]]
    fig = plot.word_clouds(importance_rows, tokens_lists, seed=0)
    assert isinstance(fig, Figure)
