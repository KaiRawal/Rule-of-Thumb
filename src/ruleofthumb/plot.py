"""Visualisation utilities for RoT explanations.

Colour convention throughout: **red = evidence toward the explained class,
blue = evidence against** (matching the legacy saliency overlays and SHAP).

Baseline semantics — how SHAP concepts translate to RoT:

- SHAP decomposes ``f(x) = phi_0 + sum_i phi_i`` with ``phi_0 = E[f(X)]``.
  RoT's surrogate is additive by construction: the class-``k`` score is
  ``s_k(x) = g_k + sum_d a_kd * (x_d + b_kd)``, and ``get_explanation``
  returns exactly the per-feature terms. The SHAP base value therefore maps
  to the **class bias ``g_k``**, and ``f(x)`` maps to the **surrogate score**
  (RoT explains its own surrogate of the black box, not the black box
  directly).
- RoT folds each feature's shift ``b`` into its contribution, so features
  are measured from zero-shift rather than mean-centred baselines.
- Text scores are length-normalised means over tokens, so token importances
  do **not** sum to the score; text visualisations show raw signed token
  weights only (highlighting and word clouds, no waterfall/force).

Tabular rendering delegates to the :mod:`shap` package: RoT values and the
class bias are packed into a :class:`shap.Explanation` and handed to
``shap.plots.waterfall`` / ``force`` / ``decision`` / ``bar`` / ``beeswarm``.
"""

from __future__ import annotations

import html as _html
from typing import Any

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


def _values_and_base(explainer, x, class_idx):
    """Signed importance values for the first sample plus the class-bias baseline."""
    importances = explainer.get_explanation(x)
    if importances.ndim >= 3:  # multiclass: (N, K, ...)
        values = importances[0, class_idx]
    else:  # binary: (N, ...) already reduced to the class-1 contributions
        values = importances[0]
    return np.asarray(values, dtype=np.float64), float(explainer.model.g[class_idx].item())


def _feature_data(x):
    """Per-feature input values of the explained sample, when available."""
    if isinstance(x, list):
        return None
    array = np.asarray(x)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array[0]
    return None


def _shap_explanation(values, base, data=None, feature_names=None):
    import shap

    kwargs = {"values": values, "base_values": base}
    if data is not None:
        kwargs["data"] = data
    if feature_names is not None:
        kwargs["feature_names"] = [str(name) for name in feature_names]
    return shap.Explanation(**kwargs)


def _current_figure():
    figure = plt.gcf()
    if not figure.axes:
        figure.add_subplot(111)
    return figure


def waterfall(explainer, x, *, feature_names=None, max_display=10, class_idx=1):
    """SHAP-style waterfall plot of one sample's RoT explanation."""
    import shap

    values, base = _values_and_base(explainer, x, class_idx)
    explanation = _shap_explanation(values, base, data=_feature_data(x), feature_names=feature_names)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    return _current_figure()


def force(explainer, x, *, feature_names=None, class_idx=1):
    """SHAP-style force plot of one sample's RoT explanation."""
    import shap

    values, base = _values_and_base(explainer, x, class_idx)
    explanation = _shap_explanation(values, base, data=_feature_data(x), feature_names=feature_names)
    shap.plots.force(explanation, matplotlib=True, show=False)
    return _current_figure()


def decision(explainer, x, *, feature_names=None, class_idx=1):
    """SHAP-style decision plot of one sample's RoT explanation."""
    import shap

    values, base = _values_and_base(explainer, x, class_idx)
    shap.plots.decision(base, values, feature_names=feature_names, show=False)
    return _current_figure()


def bar(explainer, x, *, feature_names=None, max_display=10, class_idx=1):
    """SHAP-style bar plot of mean absolute importances over a batch."""
    import shap

    importances = explainer.get_explanation(x)
    values = importances[:, class_idx] if importances.ndim >= 3 else importances
    _, base = _values_and_base(explainer, x[:1], class_idx)
    data = None if isinstance(x, list) else np.asarray(x)[: len(values)]
    explanation = _shap_explanation(np.asarray(values), np.full(len(values), base), data=data, feature_names=feature_names)
    shap.plots.bar(explanation, max_display=max_display, show=False)
    return _current_figure()


def beeswarm(explainer, x, *, feature_names=None, max_display=10, class_idx=1):
    """SHAP-style beeswarm plot of a batch's importances against feature values."""
    import shap

    importances = explainer.get_explanation(x)
    values = importances[:, class_idx] if importances.ndim >= 3 else importances
    _, base = _values_and_base(explainer, x[:1], class_idx)
    data = None if isinstance(x, list) else np.asarray(x)[: len(values)]
    explanation = _shap_explanation(np.asarray(values), np.full(len(values), base), data=data, feature_names=feature_names)
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    return _current_figure()


def _token_pairs(importance_row, tokens, max_tokens):
    importance = np.asarray(importance_row).flatten()
    pairs = [(str(token), float(value)) for token, value in zip(tokens, importance) if token]
    if max_tokens is not None:
        keep = sorted(range(len(pairs)), key=lambda i: abs(pairs[i][1]), reverse=True)[:max_tokens]
        pairs = [pairs[i] for i in sorted(keep)]
    scale = max((abs(value) for _, value in pairs), default=0.0) or 1.0
    return pairs, scale


def _token_rgba(value, scale):
    alpha = abs(value) / scale * 0.85
    if value > 0:
        return f"rgba(255, 60, 60, {alpha:.2f})"
    if value < 0:
        return f"rgba(60, 60, 255, {alpha:.2f})"
    return "transparent"


def _token_colour(value, scale):
    """Matplotlib-compatible ``(r, g, b, a)`` float tuple."""
    alpha = abs(value) / scale * 0.85
    if value > 0:
        return (1.0, 60 / 255, 60 / 255, alpha)
    if value < 0:
        return (60 / 255, 60 / 255, 1.0, alpha)
    return (1.0, 1.0, 1.0, 0.0)


def text_html(importance_row, tokens, *, max_tokens=None):
    """Token-highlighted HTML for Jupyter; red backgrounds push toward the class, blue against.

    Returns an :class:`IPython.display.HTML` object when IPython is
    installed, otherwise the raw HTML string.
    """
    pairs, scale = _token_pairs(importance_row, tokens, max_tokens)
    spans = [
        f'<span style="background-color: {_token_rgba(value, scale)}; '
        f'border-radius: 3px; padding: 1px 3px; margin: 1px;">{_html.escape(token)}</span>'
        for token, value in pairs
    ]
    markup = f'<div style="font-family: monospace; line-height: 2.0;">{" ".join(spans)}</div>'
    try:
        from IPython.display import HTML

        return HTML(markup)
    except ImportError:
        return markup


def text_matplotlib(importance_row, tokens, *, max_tokens=None, width=16):
    """Static matplotlib rendering of :func:`text_html` for image export."""
    pairs, scale = _token_pairs(importance_row, tokens, max_tokens)
    columns = max(8, int(np.sqrt(len(pairs))) * 2)
    rows = int(np.ceil(len(pairs) / columns)) or 1
    figure, axis = plt.subplots(figsize=(width, 0.6 * rows + 0.5))
    axis.set_xlim(0, columns)
    axis.set_ylim(0, rows)
    axis.axis("off")
    for index, (token, value) in enumerate(pairs):
        column, row = index % columns, rows - 1 - index // columns
        axis.text(
            column + 0.5,
            row + 0.5,
            token,
            ha="center",
            va="center",
            family="monospace",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": _token_colour(value, scale), "edgecolor": "none"},
        )
    figure.tight_layout()
    return figure


def saliency(
    heatmap: np.ndarray,
    image: np.ndarray | None = None,
    *,
    power: float = 1.0,
    trim: float = 2.0,
    size: tuple[float, float] = (5, 5),
    ax: Any | None = None,
):
    """Signed saliency overlay: red pixels push toward the class, blue against.

    Port of the legacy overlay algorithm: a sign-preserving power transform,
    independent percentile-trimmed normalisation of the positive and negative
    masses, and saturation compression above half intensity.

    Args:
        heatmap: ``(H, W)`` signed importance map (channels summed).
        image: optional ``(H, W, 3)`` RGB image to overlay onto.
        power: exponent applied to absolute magnitudes before normalisation.
        trim: percentile used to clip extremes on each side.
        size: figure size when creating a new figure.
        ax: optional existing axes to draw into.
    """
    if power <= 0:
        raise ValueError("power must be positive")
    heat = np.sign(np.asarray(heatmap, dtype=np.float32)) * np.power(np.abs(np.asarray(heatmap, dtype=np.float32)), power)
    low, high = np.percentile(heat, trim), np.percentile(heat, 100 - trim)
    positive = np.clip(heat / (high if high > 0 else 1e-8), 0.0, 1.0)
    negative = np.clip(-heat / (-low if low < 0 else 1e-8), 0.0, 1.0)
    positive = np.where(positive > 0.5, 0.5 + (positive - 0.5) * 0.8, positive)
    negative = np.where(negative > 0.5, 0.5 + (negative - 0.5) * 0.8, negative)

    blue = matplotlib.colors.LinearSegmentedColormap.from_list("transparent_blue", [(0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0)])
    red = matplotlib.colors.LinearSegmentedColormap.from_list("transparent_red", [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)])

    if ax is None:
        figure = plt.figure(figsize=size)
        ax = figure.gca()
    else:
        figure = ax.figure
    extent = [0, 1, 0, 1]
    if image is not None:
        ax.imshow(np.asarray(image), extent=extent)
    for layer, cmap in ((negative, blue), (positive, red)):
        ax.imshow(layer, cmap=cmap, origin="upper", extent=extent, interpolation="bicubic", vmin=0, vmax=1)
    ax.axis("off")
    figure.tight_layout(pad=0)
    return figure


def word_clouds(importance_rows, tokens_lists, *, stopwords=None, width=800, height=400, seed=0):
    """Positive, negative and combined word clouds from token-level explanations.

    Token weights are averaged across documents; a word appears in the red
    (positive) cloud when its mean weight is positive and in the blue
    (negative) cloud otherwise. Departs from the legacy green/red cloud
    colours for consistency with the rest of the module. One-sided
    importances render fine: the empty panel shows a "no tokens" label.
    """
    from wordcloud import STOPWORDS, WordCloud

    totals, counts = {}, {}
    for row, tokens in zip(importance_rows, tokens_lists):
        for token, value in zip(tokens, np.asarray(row).flatten()):
            if not token:
                continue
            totals[token] = totals.get(token, 0.0) + float(value)
            counts[token] = counts.get(token, 0) + 1
    means = {token: total / counts[token] for token, total in totals.items()}

    stop = set(STOPWORDS) | set(stopwords or ())
    positive = {token: weight for token, weight in means.items() if weight > 0 and token.lower() not in stop}
    negative = {token: -weight for token, weight in means.items() if weight < 0 and token.lower() not in stop}
    combined = {token: abs(weight) for token, weight in means.items() if token.lower() not in stop}

    def make(frequencies, colour_func):
        cloud = WordCloud(
            width=width,
            height=height,
            random_state=seed,
            background_color="white",
            relative_scaling=0.5,
            color_func=colour_func,
        )
        return cloud.generate_from_frequencies(frequencies)

    def fixed(colour):
        return lambda *args, **kwargs: colour

    def by_sign(word, *args, **kwargs):
        return "#d62728" if means.get(word, 0.0) >= 0 else "#1f77b4"

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (frequencies, colour_func, title, empty_label) in zip(
        axes,
        [
            (negative, fixed("#1f77b4"), "against the class", "no tokens against the class"),
            (positive, fixed("#d62728"), "toward the class", "no tokens toward the class"),
            (combined, by_sign, "combined", "no tokens"),
        ],
    ):
        if frequencies:
            axis.imshow(make(frequencies, colour_func).to_array())
        else:
            axis.text(0.5, 0.5, empty_label, ha="center", va="center", fontsize=12)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.tight_layout()
    return figure
