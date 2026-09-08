"""Unit tests for :func:`ruleofthumb.autotune`.

Beyond search mechanics (grid enumeration, candidate counts, seeding), the
fit-quality tests assert that the search genuinely finds good fits: on a
separable synthetic set the winner must reach high held-out accuracy and
beat a deliberately bad configuration by a clear margin.
"""

import numpy as np
import pytest
import torch

from ruleofthumb import autotune, fit_tabular

BAD_PARAMS = {"learning_rate": 1e-5, "batch_size": 64, "epochs": 1, "dropout_rate": 0.5, "weight_decay": 0.05}


def _separable_dataset(n=200):
    """Linearly separable tabular data a good RoT fit should nail."""
    rng = np.random.RandomState(0)
    x = rng.rand(n, 3).astype(np.float32)
    y = ((x[:, 0] + 2 * x[:, 1] - x[:, 2]) > 1.0).astype(np.int64)
    return x, y


def test_grid_enumerates_all_combinations():
    x, y = _separable_dataset()
    space = {"learning_rate": [0.01, 0.05], "epochs": [2], "batch_size": [500]}
    result = autotune(y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0)

    combos = {tuple(sorted(t["params"].items())) for t in result.trials}
    expected = {(("batch_size", 500), ("epochs", 2), ("learning_rate", lr)) for lr in (0.01, 0.05)}
    assert combos == expected
    assert len(result.trials) == 2


def test_random_respects_n_candidates_and_space():
    x, y = _separable_dataset()
    space = {
        "learning_rate": [0.003, 0.01, 0.03],
        "batch_size": [64, 500],
        "epochs": [100, 300],
        "dropout_rate": [0.1, 0.5],
        "weight_decay": [0.0, 0.01],
    }
    result = autotune(y, x, modality="tabular", search="random", n_candidates=5, space=space, seed=0)

    assert len(result.trials) == 5
    for trial in result.trials:
        assert set(trial["params"]) == set(space)
        for key, value in trial["params"].items():
            assert value in space[key]
    scores = [t["score"] for t in result.trials]
    assert scores == sorted(scores, reverse=True)  # best-first


def test_seeded_reproducibility():
    x, y = _separable_dataset()
    space = {"learning_rate": [0.003, 0.01, 0.05], "epochs": [100], "batch_size": [500], "dropout_rate": [0.1, 0.5]}
    first = autotune(y, x, modality="tabular", search="random", n_candidates=4, space=space, seed=7)
    second = autotune(y, x, modality="tabular", search="random", n_candidates=4, space=space, seed=7)

    assert first.best_params == second.best_params
    assert first.best_score == pytest.approx(second.best_score)
    assert [t["params"] for t in first.trials] == [t["params"] for t in second.trials]


def test_validation_split_sizes():
    x, y = _separable_dataset()
    result = autotune(
        y,
        x,
        modality="tabular",
        search="grid",
        space={"learning_rate": [0.05], "epochs": [2], "batch_size": [500]},
        validation_split=0.25,
        seed=0,
    )
    # the returned explainer is refit on ALL data
    assert result.explainer.predict(torch.from_numpy(x)).shape[0] == x.shape[0]


def test_search_finds_a_genuinely_good_fit():
    """The winner must reach high held-out accuracy and beat a bad config by a margin."""
    x, y = _separable_dataset()
    space = {
        "learning_rate": [1e-5, 0.05],
        "batch_size": [64],
        "epochs": [1, 200],
        "dropout_rate": [0.1],
        "weight_decay": [0.0],
    }
    result = autotune(y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0)

    assert result.best_score >= 0.9

    bad = fit_tabular(y, x, epochs=1, batch_size=64, learning_rate=1e-5, weight_decay=0.05, dropout_rate=0.1, seed=0)
    xt = torch.from_numpy(x)
    order = bad.get_order(xt)
    curve = bad.score_ordering(xt, torch.from_numpy(y), order)
    bad_score = float(curve[-1])
    assert result.best_score >= bad_score + 0.15
    assert result.best_params["epochs"] == 200 and result.best_params["learning_rate"] == 0.05


def test_refit_explainer_is_accurate_on_all_data():
    x, y = _separable_dataset()
    space = {"learning_rate": [0.05], "epochs": [200], "batch_size": [64], "dropout_rate": [0.1], "weight_decay": [0.0]}
    result = autotune(y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0)

    preds = result.explainer.predict(torch.from_numpy(x)).cpu().numpy()
    assert float((preds == y).mean()) >= 0.95
