"""Unit tests for the opt-in ``nonlinear=`` elementwise response functions."""

import numpy as np
import pytest
import torch

from ruleofthumb import fit_tabular, load_explainer
from ruleofthumb.core import RoT
from ruleofthumb.image import RoTImage
from ruleofthumb.shapes import HingeResponse, RBFResponse, resolve_nonlinear
from ruleofthumb.text import RoTText


def test_default_path_unchanged():
    model = RoT(2, (3,))
    assert model.response is None
    assert model.nonlinear_spec is None
    assert not any(key.startswith("response") for key in model.state_dict())

    explainer = fit_tabular(np.zeros(4, dtype=np.int64), np.zeros((4, 3)), epochs=1, seed=0)
    assert explainer.model.response is None


def test_resolve_nonlinear_validation():
    assert resolve_nonlinear("rbf") == ("rbf", {})
    assert resolve_nonlinear({"type": "hinge", "n_bases": 5}) == ("hinge", {"n_bases": 5})
    with pytest.raises(ValueError, match="unknown nonlinear type"):
        resolve_nonlinear("poly")
    with pytest.raises(ValueError, match="unknown nonlinear type"):
        resolve_nonlinear({"n_bases": 4})
    with pytest.raises(TypeError, match="string or a dict"):
        resolve_nonlinear(7)


@pytest.mark.parametrize("response", [RBFResponse(6), HingeResponse(6)])
def test_identity_at_init(response):
    """Zero-initialised residual coefficients make s(x) == x exactly."""
    points = torch.randn(50)
    assert torch.allclose(response(points), points, atol=1e-7)


def test_hyperparameters_reach_the_module():
    model = RoT(2, (3,), nonlinear={"type": "rbf", "n_bases": 4}, device="cpu")
    assert model.nonlinear_spec == {"type": "rbf", "n_bases": 4}
    assert model.response.centres.shape == (4,)
    assert any(key.startswith("response") for key in model.state_dict())


def test_importance_matches_manual_formula():
    x = torch.randn(5, 3)
    model = RoT(2, (3,), nonlinear="rbf", device="cpu")
    with torch.no_grad():
        model.a.copy_(torch.randn_like(model.a))
        model.b.copy_(torch.randn_like(model.b))
    expected = model.a[None] * (model.response(x)[:, None] + model.b[None])
    assert torch.allclose(model.importance(x), expected, atol=1e-6)


@pytest.mark.parametrize("spec", ["rbf", {"type": "hinge", "n_bases": 5}])
@pytest.mark.parametrize("classes", [2, 3])
def test_additive_decomposition_all_modalities(spec, classes):
    x_tab = torch.randn(6, 4)
    tab = RoT(classes, (4,), nonlinear=spec, device="cpu")
    imp = tab.importance(x_tab).detach()
    score = tab.score(x_tab)
    assert torch.allclose(score, imp.sum(-1) + tab.g[None], atol=1e-5)

    x_txt = torch.randn(6, 5, 3)
    txt = RoTText(classes, (5, 3), nonlinear=spec, device="cpu")
    imp = txt.importance(x_txt).detach()
    expected = (imp.sum(2) / 5).sum(-1) + txt.g[None]  # token-mean over real tokens, summed over dims
    assert torch.allclose(txt.score(x_txt), expected, atol=1e-5)

    x_img = torch.randn(6, 2, 4, 4)
    img = RoTImage(classes, (2,), nonlinear=spec, device="cpu")
    imp = img.importance(x_img).detach()
    assert torch.allclose(img.score(x_img), imp.sum((2, 3, 4)) + img.g[None], atol=1e-5)


def test_masked_text_positions_stay_zero():
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    model = RoTText(2, (4, 3), nonlinear="rbf", device="cpu")
    imp = model.importance(x, mask=mask)
    assert torch.all(imp[:, :, 2:, :] == 0)


def _ring_data(n, seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2, 2, size=(n, 2)).astype(np.float32)
    y = ((x**2).sum(1) < 2.0).astype(np.int64)
    return x, y


@pytest.mark.parametrize("nonlinear", ["rbf", "hinge"])
def test_nonlinear_fit_separates_ring_linear_cannot(nonlinear):
    """A circular boundary is not linearly separable; shaped models learn it."""
    x, y = _ring_data(500, seed=0)

    linear = fit_tabular(y, x, epochs=250, batch_size=500, learning_rate=0.05, seed=0, device="cpu")
    linear_accuracy = float((linear.predict(torch.from_numpy(x)).cpu().numpy() == y).mean())
    assert linear_accuracy < 0.8

    shaped = fit_tabular(y, x, epochs=400, batch_size=500, learning_rate=0.05, seed=0, device="cpu", nonlinear=nonlinear)
    shaped_accuracy = float((shaped.predict(torch.from_numpy(x)).cpu().numpy() == y).mean())
    assert shaped_accuracy >= 0.85
    assert shaped_accuracy >= linear_accuracy + 0.15
    assert shaped.model.nonlinear_spec["type"] == nonlinear


def test_reveal_pipeline_works_with_nonlinear():
    x, y = _ring_data(100, seed=1)
    explainer = fit_tabular(y, x, epochs=60, batch_size=100, learning_rate=0.05, seed=0, device="cpu", nonlinear="rbf")
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    order = explainer.get_order(xt)
    curve = explainer.score_ordering(xt, yt, order)
    assert curve.shape[0] == 3  # two features -> three reveal steps
    assert float(curve[-1]) >= float(curve[0])


def test_persistence_round_trip(tmp_path):
    x, y = _ring_data(120, seed=2)
    explainer = fit_tabular(
        y, x, epochs=80, batch_size=120, learning_rate=0.05, seed=0, device="cpu", nonlinear={"type": "rbf", "n_bases": 8}
    )
    path = tmp_path / "explainer.pt"
    explainer.save(path)

    restored = load_explainer(path)
    assert restored.model.nonlinear_spec == {"type": "rbf", "n_bases": 8}
    assert np.allclose(restored.get_explanation(x), explainer.get_explanation(x), atol=1e-6)
