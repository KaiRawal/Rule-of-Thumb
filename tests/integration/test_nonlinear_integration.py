"""Integration test: non-linear additive RoT on committed real data."""

import numpy as np
import torch
from _helpers import rot_accuracy

from ruleofthumb import fit_tabular

SEED = 0


def test_nonlinear_tabular_matches_or_beats_linear_on_wine(wine):
    """The shaped surrogate must not hurt fidelity on real multiclass data."""
    x, y = wine["x"], wine["y_gbm"]

    linear = fit_tabular(y, x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED, n_classes=3)
    shaped = fit_tabular(y, x, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED, n_classes=3, nonlinear="rbf")

    linear_accuracy = rot_accuracy(linear, x, y)
    shaped_accuracy = rot_accuracy(shaped, x, y)
    assert shaped_accuracy >= 0.85
    assert shaped_accuracy >= linear_accuracy - 0.02

    imp = shaped.get_explanation(x)
    assert imp.shape == (len(x), 3, x.shape[1])
    assert np.isfinite(imp).all()

    # explanations stay exactly additive with the class biases
    scores = shaped.model.score(torch.from_numpy(x)).detach().cpu().numpy()
    biases = shaped.model.g.detach().cpu().numpy()
    assert np.allclose(imp.sum(-1) + biases[None], scores, atol=1e-3)
