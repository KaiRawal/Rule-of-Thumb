"""Integration tests: tabular modality against real data and models.

Binary: breast-cancer features explained through a fitted LogisticRegression.
Multiclass: flattened digit images (10 classes) through a fitted RandomForest.

All black boxes are committed artifacts; only the RoT explainer is fitted
live, exercising the :func:`ruleofthumb.fit_tabular` facade end to end.
"""

import numpy as np
import torch

from ruleofthumb import fit_tabular

SEED = 0


def _fit_binary(x, y):
    return fit_tabular(y, x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED)


def test_binary_explanation_shape_additivity_and_fidelity(tabular_binary):
    x, y = tabular_binary["x"], tabular_binary["y"]
    exp = _fit_binary(x, y)
    assert exp.modality == "tabular"
    assert exp.model.classes == 2

    imp = exp.get_explanation(x)
    assert imp.shape == x.shape
    assert np.isfinite(imp).all()

    # additivity: class-1 contributions + class-1 bias reproduce the score
    scores = exp.model.score(torch.from_numpy(x)).detach().cpu().numpy()
    bias1 = exp.model.g[1].detach().cpu().numpy()
    assert np.allclose(imp.sum(1) + bias1, scores[:, 1], atol=1e-3)

    # fidelity: the surrogate reproduces the black-box labels
    surrogate_predictions = exp.predict(torch.from_numpy(x)).cpu().numpy()
    accuracy = float((surrogate_predictions == y).mean())
    assert accuracy >= 0.85  # black box itself achieves ~0.988 on this data


def test_binary_top_features_match_logistic_coefficients(tabular_binary):
    x, y, coef = tabular_binary["x"], tabular_binary["y"], tabular_binary["coef"]
    exp = _fit_binary(x, y)
    imp = exp.get_explanation(x)

    k = 8
    top_black_box = set(np.argsort(-np.abs(coef))[:k])
    top_surrogate = set(np.argsort(-np.abs(imp).mean(0))[:k])
    assert len(top_black_box & top_surrogate) >= k // 2

    # tighter anchors: the importance profile tracks the coefficient profile
    assert len(set(np.argsort(-np.abs(coef))[:7]) & set(np.argsort(-np.abs(imp).mean(0))[:7])) >= 4
    assert np.corrcoef(np.abs(imp).mean(0), np.abs(coef))[0, 1] >= 0.5


def test_binary_reveal_curve_recovers_full_accuracy(tabular_binary):
    x, y = tabular_binary["x"], tabular_binary["y"]
    exp = _fit_binary(x, y)
    xt = torch.from_numpy(x)
    order = exp.get_order(xt)
    assert order.shape == x.shape
    # every feature ranked exactly once per sample
    assert np.array_equal(np.sort(order, axis=1), np.tile(np.arange(x.shape[1]), (len(x), 1)))

    curve = exp.score_ordering(xt, torch.from_numpy(y.astype(np.int64)), order)
    assert curve.shape == (x.shape[1] + 1,)
    # revealing all features equals the plain surrogate accuracy
    full_accuracy = float((exp.predict(xt).cpu().numpy() == y).mean())
    assert abs(float(curve[-1]) - full_accuracy) < 1e-6
    # most-important-first beats starting from the bias alone by a wide margin
    assert curve[-1] >= curve[0] + 0.3


def test_binary_seed_reproducibility(tabular_binary):
    x, y = tabular_binary["x"], tabular_binary["y"]
    imp_a = _fit_binary(x, y).get_explanation(x)
    imp_b = _fit_binary(x, y).get_explanation(x)
    assert np.allclose(imp_a, imp_b)


def test_multiclass_explanation_shape_and_fidelity(tabular_multiclass):
    x, y = tabular_multiclass["x"], tabular_multiclass["y"]
    exp = fit_tabular(y, x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED, n_classes=10)
    imp = exp.get_explanation(x)
    assert imp.shape == (len(x), 10, x.shape[1])
    for k in range(10):
        assert (imp[:, k, :] != 0).any()

    predictions = exp.predict(torch.from_numpy(x)).cpu().numpy()
    accuracy = float((predictions == y).mean())
    assert accuracy >= 0.5  # far above the 10% chance line


def test_multiclass_confusion_counts_match_active_samples(tabular_multiclass):
    x, y = tabular_multiclass["x"], tabular_multiclass["y"]
    exp = fit_tabular(y, x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED, n_classes=10)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.int64))
    order = exp.get_order(xt)

    confusion = exp.score_ordering(xt, yt, order, return_confusion=True)
    pred = exp.ordered_predict(xt, order).cpu()
    valid = pred != -1
    active = valid.any(0)
    steps = int(active.nonzero().max()) + 1
    assert tuple(confusion.shape) == (steps, 10, 10)
    for j in range(steps):
        n_active = int(valid[:, j].sum())
        assert int(confusion[j].sum()) == n_active

    # final-step diagonal matches the plain multiclass accuracy
    final_correct = sum(confusion[steps - 1][c][c].item() for c in range(10))
    assert abs(final_correct / len(x) - float((exp.predict(xt).cpu().numpy() == y).mean())) < 1e-6
