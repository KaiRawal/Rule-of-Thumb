"""Unit tests for :func:`rot.autotune`.

Beyond search mechanics (grid enumeration, candidate counts, seeding), the
fit-quality tests assert that the search genuinely finds good fits: on a
separable synthetic set the winner must reach high held-out accuracy and
beat a deliberately bad configuration by a clear margin.
"""

import numpy as np
import pytest
import torch

from ruleofthumb import autotune, fit_tabular

BAD_PARAMS = {"learning_rate": 1e-5, "batch_size": 64, "epochs": 1, "weight_decay": 0.05}


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
    space = {"learning_rate": [0.003, 0.01, 0.05], "epochs": [100], "batch_size": [500]}
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
    """The winner must reach high held-out agreement and beat a bad config by a margin."""
    x, y = _separable_dataset()
    space = {
        "learning_rate": [1e-5, 0.05],
        "batch_size": [64],
        "epochs": [1, 200],
        "weight_decay": [0.0],
    }
    result = autotune(y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0)

    assert result.best_score >= 0.85

    bad = fit_tabular(y, x, epochs=1, batch_size=64, learning_rate=1e-5, weight_decay=0.05, seed=0, tune=None)
    bad_preds = np.asarray(bad.predict(torch.from_numpy(x)).cpu())
    bad_score = float((bad_preds == y).mean())
    assert result.best_score >= bad_score + 0.15
    assert result.best_params["epochs"] == 200 and result.best_params["learning_rate"] == 0.05


def test_reveal_scoring_is_opt_in():
    """Reveal-based candidate scoring runs only when scoring="reveal" is named."""
    x, y = _separable_dataset()
    space = {
        "learning_rate": [1e-5, 0.05],
        "batch_size": [64],
        "epochs": [1, 200],
        "weight_decay": [0.0],
    }
    result = autotune(
        y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0, scoring="reveal"
    )

    assert result.best_score >= 0.85
    assert result.best_params["epochs"] == 200 and result.best_params["learning_rate"] == 0.05


def test_unknown_scoring_raises():
    x, y = _separable_dataset()
    with pytest.raises(ValueError, match="unknown scoring"):
        autotune(y, x, modality="tabular", search="grid", space={"learning_rate": [0.05]}, seed=0, scoring="auc")


def test_default_tuning_runs_no_reveal_machinery(monkeypatch):
    """Default scoring must not invoke the reveal pipeline; opt-in scoring must."""
    from ruleofthumb.explain import Explainer

    def _forbidden(*args, **kwargs):
        raise AssertionError("reveal machinery must not run unless scoring='reveal' is named")

    for name in ("get_order", "ordered_predict", "score_ordering"):
        monkeypatch.setattr(Explainer, name, _forbidden)

    x, y = _separable_dataset()
    space = {"learning_rate": [0.05], "epochs": [4], "batch_size": [64]}
    result = autotune(y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0)
    assert np.isfinite(result.best_score)

    with pytest.raises(AssertionError, match="reveal machinery"):
        autotune(
            y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0, scoring="reveal"
        )


def test_refit_explainer_is_accurate_on_all_data():
    x, y = _separable_dataset()
    space = {"learning_rate": [0.05], "epochs": [200], "batch_size": [64], "weight_decay": [0.0]}
    result = autotune(y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0)

    preds = result.explainer.predict(torch.from_numpy(x)).cpu().numpy()
    assert float((preds == y).mean()) >= 0.9


def _multiclass_dataset(n=150):
    """Three-class tabular data (sandbox Bug 1 repro, scaled down)."""
    rng = np.random.RandomState(7)
    x = rng.rand(n, 5).astype(np.float32)
    logits = np.stack(
        [1.5 * x[:, 0] - 0.5 * x[:, 1], -x[:, 0] + 1.5 * x[:, 2], 1.2 * x[:, 3] - 0.4 * x[:, 4]], axis=1
    )
    return x, np.argmax(logits, axis=1).astype(np.int64)


def test_autotune_infers_n_classes_for_multiclass():
    x, y = _multiclass_dataset()
    space = {"learning_rate": [0.05], "epochs": [4], "batch_size": [500]}
    result = autotune(y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0)

    assert result.explainer.model.classes == 3
    assert np.isfinite(result.best_score)
    preds = result.explainer.predict(torch.from_numpy(x)).cpu().numpy()
    assert preds.shape == (x.shape[0],)


def test_autotune_explicit_n_classes_override():
    x, y = _multiclass_dataset()
    space = {"learning_rate": [0.05], "epochs": [4], "batch_size": [500]}
    result = autotune(
        y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0, n_classes=3
    )

    assert result.explainer.model.classes == 3


def test_dropout_rate_is_not_tunable():
    """Dropout is a fixed method constant (0.5): no factory, model or search space accepts it."""
    import inspect

    from ruleofthumb import RoT, fit_image, fit_text
    from ruleofthumb.image import RoTImage
    from ruleofthumb.text import RoTText

    for fn in (RoT, RoTText, RoTImage, fit_tabular, fit_text, fit_image, autotune):
        assert "dropout_rate" not in inspect.signature(fn).parameters
    x, y = _separable_dataset()
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        autotune(
            y,
            x,
            modality="tabular",
            search="grid",
            space={"learning_rate": [0.05], "epochs": [2], "batch_size": [500], "dropout_rate": [0.5]},
            seed=0,
        )


def test_autotune_forwards_model_kwargs():
    x, y = _separable_dataset()
    space = {"learning_rate": [0.05], "epochs": [4], "batch_size": [500]}
    result = autotune(
        y, x, modality="tabular", search="grid", space=space, validation_split=0.25, seed=0, nonlinear="hinge"
    )

    assert result.explainer.model.nonlinear_spec is not None
    assert set(result.best_params) == set(space)


def test_default_space_is_modality_aware():
    from ruleofthumb.tune import DEFAULT_SPACES

    assert "l1_penalty" in DEFAULT_SPACES["text"]  # text can search it
    assert "l1_penalty" not in DEFAULT_SPACES["tabular"]  # tabular factory has no such knob
    x, y = _separable_dataset()
    with pytest.raises(ValueError, match="unknown hyperparameters in space for tabular"):
        autotune(y, x, modality="tabular", space={"l1_penalty": [0.0]}, seed=0)


def test_oom_candidate_is_skipped_and_search_survives(monkeypatch):
    from ruleofthumb import tune

    x, y = _separable_dataset()
    good = tune._FACTORIES["tabular"]

    def flaky(*args, **kwargs):  # lr 0.1 candidates OOM; the rest fit fine
        if kwargs.get("learning_rate") == 0.1:
            raise RuntimeError("CUDA out of memory")
        return good(*args, **kwargs)

    monkeypatch.setitem(tune._FACTORIES, "tabular", flaky)
    space = {"learning_rate": [0.05, 0.1], "epochs": [100], "batch_size": [64]}
    result = autotune(y, x, modality="tabular", search="grid", space=space, seed=0)

    assert any(t["score"] == float("-inf") and "error" in t for t in result.trials)  # OOM one recorded, not fatal
    assert result.best_params["learning_rate"] == 0.05  # the good one wins


def test_non_oom_error_propagates(monkeypatch):
    from ruleofthumb import tune

    def bad_kwarg(*args, **kwargs):
        raise TypeError("unexpected keyword")  # a real bug, not OOM: must not be swallowed

    monkeypatch.setitem(tune._FACTORIES, "tabular", bad_kwarg)
    x, y = _separable_dataset()
    with pytest.raises(TypeError, match="unexpected keyword"):
        autotune(y, x, modality="tabular", search="grid",
                 space={"learning_rate": [0.05], "epochs": [100], "batch_size": [64]}, seed=0)


def test_all_candidates_oom_raises(monkeypatch):
    from ruleofthumb import tune

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setitem(tune._FACTORIES, "tabular", oom)
    x, y = _separable_dataset()
    with pytest.raises(RuntimeError, match="every autotune candidate failed"):
        autotune(y, x, modality="tabular", search="grid",
                 space={"learning_rate": [0.05], "epochs": [100], "batch_size": [64]}, seed=0)


def test_successive_halving_prunes_but_keeps_the_winner():
    x, y = _separable_dataset()
    space = {"learning_rate": [1e-5, 1e-4, 0.03, 0.05], "epochs": [60], "batch_size": [64]}  # >=4 -> halving on
    result = autotune(y, x, modality="tabular", search="grid", space=space, seed=0)

    assert any(t.get("pruned") for t in result.trials)      # the cheap screen abandoned the weak half
    assert result.best_score > 0.9                           # a good learning rate still won
    assert result.best_params["learning_rate"] in (0.03, 0.05)


def test_tune_flag_attaches_params_only_when_tuning():
    x, y = _separable_dataset()
    tuned = fit_tabular(y, x, tune=True, n_classes=2)
    assert tuned.best_params_ is not None and tuned.best_score_ is not None
    fixed = fit_tabular(y, x, tune=None, n_classes=2)
    assert not hasattr(fixed, "best_params_")  # the escape hatch attaches nothing


def test_tune_none_and_false_give_the_same_fixed_fit():
    x, y = _separable_dataset()
    a = fit_tabular(y, x, tune=None, epochs=50, learning_rate=0.05, seed=0, n_classes=2)
    b = fit_tabular(y, x, tune=False, epochs=50, learning_rate=0.05, seed=0, n_classes=2)
    assert np.allclose(a.get_explanation(x), b.get_explanation(x))


def test_bad_tune_value_raises():
    x, y = _separable_dataset()
    with pytest.raises(TypeError, match="tune must be"):
        fit_tabular(y, x, tune="yes", n_classes=2)


def test_recursion_guard_candidates_fit_with_tuning_off():
    from ruleofthumb import tune

    assert tune._FACTORIES["tabular"] is fit_tabular  # autotune calls the real factory with tune=None
    x, y = _separable_dataset()
    result = fit_tabular(y, x, tune=True, n_classes=2)  # would recurse forever without the tune=None guard
    assert result.best_params_ is not None


def test_plateau_stop_reaches_fit_through_factory_and_tune():
    """patience= reaches the fit through the factory (fixed path) and through
    autotune (refit), trimming below the epoch cap — fit-to-convergence (ToDo 29)."""
    x, y = _separable_dataset()
    full = fit_tabular(y, x, n_classes=2, seed=0, epochs=400, tune=None, patience=None)
    stopped = fit_tabular(y, x, n_classes=2, seed=0, epochs=400, tune=None, patience=3)
    assert len(full.model.training_loss) == 400  # default off: runs the cap
    assert len(stopped.model.training_loss) < 400  # fixed path plateau-stops

    tuned = fit_tabular(y, x, n_classes=2, seed=0, patience=3,
                        tune={"search": "grid", "space": {"learning_rate": [0.05], "epochs": [400], "batch_size": [500]}})
    assert tuned.best_params_["epochs"] == 400
    assert len(tuned.model.training_loss) < 400  # refit plateau-stops too
