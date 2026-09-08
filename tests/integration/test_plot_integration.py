"""Integration tests: visualisations on real committed artifacts.

Every plot family renders from genuine explanations (breast-cancer LR
profile, digits RandomForest, TinyCNN digit images, SST-2 token importances,
pet saliency reference heatmaps) and exports to PNG bytes. Coverage spans
modalities (tabular / text / image), class counts (binary / multiclass),
rendering backends (SHAP figures, IPython HTML, matplotlib) and export paths.
"""

import io

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ruleofthumb import embed_texts, fit_image, fit_tabular, fit_text, load_images, plot

SEED = 0


def _png_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def test_tabular_plots_render_and_export(tabular_binary):
    x = tabular_binary["x"]
    explainer = fit_tabular(tabular_binary["y"], x, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)
    names = [f"feature_{i}" for i in range(x.shape[1])]

    for name in ("waterfall", "force", "decision"):
        png = _png_bytes(getattr(plot, name)(explainer, x[:1], feature_names=names))
        assert len(png) > 1000
    for name in ("bar", "beeswarm"):
        png = _png_bytes(getattr(plot, name)(explainer, x[:50], feature_names=names))
        assert len(png) > 1000


def test_text_html_highlights_real_tokens(text_sst2):
    texts = text_sst2["texts"]
    embedded = embed_texts(texts)
    mask = embedded.attention_mask
    explainer = fit_text(text_sst2["y"], embedded.embeddings, attention_mask=mask, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)

    imp = explainer.get_explanation(embedded.embeddings, attention_mask=mask)[0]
    tokens = embedded.tokens[0]
    html = getattr(plot.text_html(imp, tokens), "data", "")

    top_tokens = [tokens[i] for i in np.argsort(-np.abs(imp))[:5] if tokens[i]]
    assert all(token in html for token in top_tokens)


def test_saliency_renders_reference_heatmaps_over_pet_images(pets):
    """The committed reference heatmaps overlay cleanly onto the raw JPEGs."""
    labels = pets["labels"]
    paths = [f"{pets['images_dir']}/{name}" for name in labels["filename"]]
    images = load_images(paths, size=(224, 224))

    for i in range(min(3, len(paths))):
        fig = plot.saliency(pets["reference"][i].astype(np.float32), image=images.images[i].transpose(1, 2, 0))
        assert len(_png_bytes(fig)) > 1000


def test_word_clouds_render_review_tokens(text_sst2):
    texts = text_sst2["texts"]
    embedded = embed_texts(texts)
    mask = embedded.attention_mask
    explainer = fit_text(text_sst2["y"], embedded.embeddings, attention_mask=mask, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)

    rows, token_lists = [], []
    for i in range(len(texts)):
        imp = explainer.get_explanation(embedded.embeddings, attention_mask=mask)[i]
        rows.append(imp)
        token_lists.append([t for t, real in zip(embedded.tokens[i], mask[i]) if real])

    fig = plot.word_clouds(rows, token_lists, seed=SEED)
    assert len(_png_bytes(fig)) > 1000


def test_multiclass_tabular_plots_per_class(tabular_multiclass):
    """Multiclass SHAP plots select the right class slice; baselines differ per class."""
    x, y = tabular_multiclass["x"], tabular_multiclass["y"]
    explainer = fit_tabular(y, x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED, n_classes=10)
    names = [f"pixel_{i}" for i in range(x.shape[1])]

    # the SHAP baseline maps to the per-class bias g_k: classes must disagree
    bases = [float(explainer.model.g[k].item()) for k in (0, 5)]
    assert bases[0] != pytest.approx(bases[1])

    for name, class_idx in (("waterfall", 0), ("force", 5), ("decision", 3)):
        png = _png_bytes(getattr(plot, name)(explainer, x[:1], feature_names=names, class_idx=class_idx))
        assert len(png) > 1000
    for name in ("bar", "beeswarm"):
        png = _png_bytes(getattr(plot, name)(explainer, x[:50], feature_names=names, class_idx=7))
        assert len(png) > 1000


def test_image_saliency_from_fitted_explainer(image_multiclass):
    """Saliency overlays come from freshly fitted RoT explainers, both class counts."""
    x = image_multiclass["x"]
    backdrop = x[0, 0]  # first channel as a greyscale backdrop

    binary = fit_image(image_multiclass["y_binary"], x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED, n_classes=2)
    imp_binary = binary.get_explanation(x)  # (N, H, W): channels already summed
    assert len(_png_bytes(plot.saliency(imp_binary[0], image=backdrop))) > 1000

    multi = fit_image(image_multiclass["y"], x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED, n_classes=10)
    imp_multi = multi.get_explanation(x)  # (N, K, H, W)
    assert len(_png_bytes(plot.saliency(imp_multi[0, 3], image=backdrop))) > 1000


def test_text_native_string_pipeline_and_matplotlib_export(text_sst2):
    """Raw strings flow end-to-end into HTML and matplotlib token plots."""
    texts, y = text_sst2["texts"], text_sst2["y"]
    explainer = fit_text(y, texts, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)

    embedded = embed_texts([texts[0]])
    tokens = [t for t, real in zip(embedded.tokens[0], embedded.attention_mask[0]) if real]
    imp_row = explainer.get_explanation([texts[0]])[0]

    html = plot.text_html(imp_row, tokens)
    assert hasattr(html, "data")  # IPython.display.HTML when IPython is importable
    top_tokens = [tokens[i] for i in np.argsort(-np.abs(imp_row[: len(tokens)]))[:5]]
    assert all(token in html.data for token in top_tokens)

    png = _png_bytes(plot.text_matplotlib(imp_row, tokens))
    assert len(png) > 1000
