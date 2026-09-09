"""Calibrated ground-truth fixtures (xai-units lesson, zero new dependencies).

Instead of asserting shapes and monotonicity on arbitrary synthetic rules,
these tests explain a *known mechanism* — an exact handcrafted linear model
in the style of XAI-Units' ``ContinuousFeaturesNN`` (ReLU-split ``w.x``,
no training, deterministic; cf. Lee et al., FAccT'25) — and assert the
explanations recover the ground-truth weights by rank, top-k and sign.
"""

import numpy as np
import pytest
import torch

from ruleofthumb import fit_tabular

CALIBRATED_W = torch.tensor([2.0, -1.5, 1.0, -0.5, 0.25, 0.0])


class _ExactLinear(torch.nn.Module):
    """Handcrafted exact linear black box: forward(x) == x @ weights, no training."""

    def __init__(self, weights):
        super().__init__()
        n = len(weights)
        self.fc0 = torch.nn.Linear(n, 2 * n, bias=False)
        self.fc1 = torch.nn.Linear(2 * n, 1, bias=False)
        with torch.no_grad():
            self.fc0.weight.zero_()
            self.fc0.weight[:n].copy_(torch.diag(weights))
            self.fc0.weight[n:].copy_(-torch.diag(weights))
            self.fc1.weight.zero_()
            self.fc1.weight[0, :n] = 1.0
            self.fc1.weight[0, n:] = -1.0

    def forward(self, x):
        return self.fc1(torch.relu(self.fc0(x))).squeeze(-1)


def _spearman(a, b):
    """Rank correlation via numpy only (no scipy in unit tests)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


@pytest.fixture
def calibrated_binary():
    rng = np.random.RandomState(0)
    x = rng.randn(256, 6).astype(np.float32)
    with torch.no_grad():
        y = (_ExactLinear(CALIBRATED_W)(torch.from_numpy(x)) > 0).numpy().astype(np.int64)
    return x, y


def test_black_box_is_exact(calibrated_binary):
    x, y = calibrated_binary
    with torch.no_grad():
        cont = _ExactLinear(CALIBRATED_W)(torch.from_numpy(x)).numpy()
    assert np.allclose(cont, x @ CALIBRATED_W.numpy())
    assert set(np.unique(y)) == {0, 1}


def test_calibrated_rank_recovery(calibrated_binary):
    x, y = calibrated_binary
    exp = fit_tabular(y, x, epochs=200, batch_size=64, learning_rate=0.05, seed=0)
    imp = exp.get_explanation(x)
    scores = np.abs(imp).mean(0)
    truth = torch.abs(CALIBRATED_W).numpy()

    assert _spearman(scores, truth) >= 0.9
    assert set(np.argsort(-scores)[:3]) == {0, 1, 2}
    assert int(np.argmin(scores)) == 5  # decoy zero weight ranks last
    for d in range(5):  # per-sample sign structure matches the mechanism w_d * x_d
        corr = np.corrcoef(imp[:, d], x[:, d])[0, 1]
        assert corr * np.sign(CALIBRATED_W[d].item()) > 0.9


def test_calibrated_multiclass_rank_recovery():
    w3 = torch.tensor(
        [
            [2.0, 0.3, -0.2, 0.1, 0.0, 0.1],
            [0.2, 1.8, 0.3, -0.1, 0.1, 0.0],
            [-0.1, 0.2, 1.6, 0.2, -0.1, 0.0],
        ]
    )
    rng = np.random.RandomState(1)
    x = rng.randn(256, 6).astype(np.float32)
    y = np.argmax(x @ w3.numpy().T, axis=1).astype(np.int64)
    exp = fit_tabular(y, x, epochs=200, batch_size=64, learning_rate=0.05, seed=0, n_classes=3)

    imp = exp.get_explanation(x)
    assert imp.shape == (256, 3, 6)
    for k in range(3):
        assert int(np.argmax(np.abs(imp[:, k, :]).mean(0))) == k


def test_interacting_trips_quality_wire():
    """Pairwise-product labels defeat the linear surrogate: the tripwire must fire, not accuracy."""
    rng = np.random.RandomState(8)
    x = rng.rand(500, 5).astype(np.float32)
    y = ((x[:, 0] > 0.5) ^ (x[:, 1] > 0.5)).astype(np.int64)
    with pytest.warns(UserWarning, match="train agreement"):
        exp = fit_tabular(y, x, epochs=30, batch_size=64, learning_rate=0.05, seed=0)
    assert exp.train_agreement_ < 0.75
