"""Integration tests: typical tabular black boxes on real datasets.

COMPAS two-year recidivism (binary, ProPublica filters) and the sklearn wine
dataset (3 classes), each explained through GradientBoosting, SVC(RBF) and
MLPClassifier black boxes fitted once by the artifact generator.

Every case asserts the RoT surrogate's own predicted-class accuracy, explicit
feature-importance anchors (``priors_count`` for COMPAS; shared top features
for wine), explanation shapes/additivity and reveal-curve endpoints.
"""

import numpy as np
import pytest
import torch
from _helpers import rot_accuracy

from ruleofthumb import fit_tabular

MODELS = ("gbm", "svc", "mlp")

# floors calibrated against the committed artifacts (see manifest.json)
ACCURACY_FLOORS = {
    "compas": {"gbm": 0.75, "svc": 0.85, "mlp": 0.75},
    "wine": {"gbm": 0.9, "svc": 0.9, "mlp": 0.9},
}


def _fit(stem, x, y):
    n_classes = 3 if stem == "wine" else 2
    return fit_tabular(y, x, epochs=300, batch_size=5000, learning_rate=0.05, seed=0, n_classes=n_classes)


@pytest.mark.parametrize("stem", ["compas", "wine"])
@pytest.mark.parametrize("model", MODELS)
def test_rot_accuracy_and_explanations_per_black_box(stem, model, request):
    data = request.getfixturevalue(stem)
    x, y = data["x"], data[f"y_{model}"]
    exp = _fit(stem, x, y)

    # the surrogate's predicted classes must reproduce the black box
    accuracy = rot_accuracy(exp, x, y)
    assert accuracy >= ACCURACY_FLOORS[stem][model]

    if stem == "compas":
        imp = exp.get_explanation(x)  # binary: class-1 contributions
        assert imp.shape == x.shape
        scores = exp.model.score(torch.from_numpy(x)).detach().cpu().numpy()
        bias1 = exp.model.g[1].detach().cpu().numpy()
        assert np.allclose(imp.sum(1) + bias1, scores[:, 1], atol=1e-3)

        # criminological anchor: priors_count drives every black box's explanation
        top2 = np.argsort(-np.abs(imp).mean(0))[:2]
        assert data["feature_names"].index("priors_count") in top2
    else:
        imp = exp.get_explanation(x)  # multiclass: full per-class output
        assert imp.shape == (len(x), 3, x.shape[1])
        for k in range(3):
            assert (imp[:, k, :] != 0).any()


def test_wine_models_share_the_same_dominant_features(wine):
    """The three black boxes' explanations agree on the dominant chemical drivers."""
    x = wine["x"]
    top3 = {}
    for model in MODELS:
        exp = _fit("wine", x, wine[f"y_{model}"])
        imp = exp.get_explanation(x)
        top3[model] = set(np.argsort(-np.abs(imp).mean((0, 1)))[:3])
    assert len(top3["gbm"] & top3["svc"]) >= 2
    assert len(top3["gbm"] & top3["mlp"]) >= 2
    assert len(top3["svc"] & top3["mlp"]) >= 2


@pytest.mark.parametrize("stem", ["compas", "wine"])
def test_reveal_curve_endpoint_equals_full_accuracy(stem, request):
    data = request.getfixturevalue(stem)
    x, y = data["x"], data["y_gbm"]
    exp = _fit(stem, x, y)
    xt = torch.from_numpy(x)
    order = exp.get_order(xt)
    curve = exp.score_ordering(xt, torch.from_numpy(y.astype(np.int64)), order)
    full_accuracy = rot_accuracy(exp, x, y)
    assert abs(float(curve[-1]) - full_accuracy) < 1e-6


def test_wine_seed_reproducibility(wine):
    x, y = wine["x"], wine["y_svc"]
    imp_a = _fit("wine", x, y).get_explanation(x)
    imp_b = _fit("wine", x, y).get_explanation(x)
    assert np.allclose(imp_a, imp_b)
