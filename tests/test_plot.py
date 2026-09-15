"""Unit tests for :mod:`rot.plot` (Agg backend, no display)."""

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from _mock_weights import data_plot_tabular, data_plot_tabular_multi, load_mock

from ruleofthumb import plot


@pytest.fixture()
def tabular_case():
    x, _, _ = data_plot_tabular()
    yield load_mock("plot_tabular"), x
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


def test_word_clouds_single_sign_renders():
    """Bug 11: one-sided importances must not crash the empty panel."""
    tokens_lists = [["t0", "t1", "t2", "t3", "t4"]]
    all_positive = [np.array([0.04, 0.05, 0.04, 0.04, 0.03])]
    assert isinstance(plot.word_clouds(all_positive, tokens_lists, seed=0), Figure)
    all_negative = [-row for row in all_positive]
    assert isinstance(plot.word_clouds(all_negative, tokens_lists, seed=0), Figure)
    plt.close("all")


@pytest.mark.parametrize("power,trim", [(0.5, 1.0), (1.0, 2.0), (2.0, 5.0)])
def test_saliency_parameter_sweep_renders(power, trim):
    rng = np.random.RandomState(0)
    heat = rng.randn(8, 8).astype(np.float32)
    assert isinstance(plot.saliency(heat, power=power, trim=trim), Figure)
    plt.close("all")


def test_reveal_single_curve_returns_figure():
    fig = plot.reveal(np.array([0.5, 0.7, 0.9]))
    assert isinstance(fig, Figure)
    plt.close("all")


def test_reveal_dict_overlay_labels_both_curves():
    fig = plot.reveal(
        {"RoT order": np.array([0.5, 0.8, 1.0]), "Random": np.array([0.5, 0.55, 0.6])},
        title="payoff",
    )
    assert isinstance(fig, Figure)
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert labels == ["RoT order", "Random"]
    plt.close("all")


def test_reveal_accepts_torch_and_list_with_labels():
    import torch

    fig = plot.reveal(
        [torch.tensor([0.4, 0.6]), np.array([0.5, 0.55])],
        labels=["rot", "random"],
    )
    assert isinstance(fig, Figure)
    plt.close("all")


def test_reveal_rejects_empty_and_mismatched_labels():
    with pytest.raises(ValueError, match="at least one curve"):
        plot.reveal([])
    with pytest.raises(ValueError, match="at least one curve"):
        plot.reveal({})
    with pytest.raises(ValueError, match="empty"):
        plot.reveal(np.array([]))
    with pytest.raises(ValueError, match="one label per curve"):
        plot.reveal([np.array([0.5]), np.array([0.6])], labels=["only-one"])
    plt.close("all")


def test_reveal_draws_into_existing_axes():
    fig, ax = plt.subplots()
    out = plot.reveal(np.array([0.5, 1.0]), ax=ax)
    assert out is fig and len(ax.lines) == 1
    plt.close("all")


@pytest.mark.parametrize("class_idx", [0, 1, 2])
def test_multiclass_per_class_plots_render(class_idx):
    x, _, _ = data_plot_tabular_multi()
    exp = load_mock("plot_tabular_multi")
    assert isinstance(plot.bar(exp, x, class_idx=class_idx), Figure)
    assert isinstance(plot.waterfall(exp, x[:1], class_idx=class_idx), Figure)
    plt.close("all")
