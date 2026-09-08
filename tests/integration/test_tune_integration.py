"""Integration tests: autotuning on real committed artifacts.

Each case runs a small candidate search and asserts the winner genuinely
fits: held-out validation accuracy must clear the same calibrated floors the
fixed-hyperparameter integration tests use, and the returned full-data refit
must reproduce that fidelity on all data.
"""

from _helpers import rot_accuracy

from ruleofthumb import autotune

SEED = 0


def test_tabular_autotune_reaches_high_fidelity(tabular_binary):
    x, y = tabular_binary["x"], tabular_binary["y"]
    space = {
        "learning_rate": [0.01, 0.05],
        "batch_size": [64, 500],
        "epochs": [150],
        "dropout_rate": [0.1, 0.5],
        "weight_decay": [0.0],
    }
    result = autotune(y, x, modality="tabular", search="random", n_candidates=3, space=space, seed=SEED)

    assert result.best_score >= 0.9  # held-out fidelity on breast cancer
    # the returned explainer is refit on all data and stays accurate
    assert rot_accuracy(result.explainer, x, y) >= 0.9


def test_text_autotune_reaches_sentiment_fidelity(text_sst2):
    x, y = text_sst2["embeddings"], text_sst2["y"]
    space = {
        "learning_rate": [0.03, 0.05],
        "batch_size": [500],
        "epochs": [100, 200],
        "dropout_rate": [0.3, 0.5],
        "weight_decay": [0.0],
    }
    result = autotune(y, x, modality="text", search="random", n_candidates=3, space=space, seed=SEED)

    assert result.best_score >= 0.85  # matches the fixed-fit floor in test_text_integration
    assert rot_accuracy(result.explainer, x, y) >= 0.85


def test_image_autotune_reaches_binary_fidelity(image_multiclass):
    x, y = image_multiclass["x"], image_multiclass["y_binary"]
    space = {
        "learning_rate": [0.03, 0.05],
        "batch_size": [500, 5000],
        "epochs": [150, 300],
        "dropout_rate": [0.3, 0.5],
        "weight_decay": [0.0],
    }
    result = autotune(y, x, modality="image", search="random", n_candidates=3, space=space, seed=SEED)

    assert result.best_score >= 0.9  # dense-vs-sparse is expressible by the surrogate
    assert rot_accuracy(result.explainer, x, y) >= 0.9
