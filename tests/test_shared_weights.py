import numpy as np
import pytest
import torch

from ruleofthumb import fit_image, fit_text, load_explainer
from ruleofthumb.image import RoTImage
from ruleofthumb.text import RoTText


def test_unshared_text_weights_are_position_specific():
    model = RoTText(2, (3, 2), share_weights=False)
    assert tuple(model.a.shape) == (2, 3, 2)
    assert tuple(model.b.shape) == (2, 3, 2)

    with torch.no_grad():
        model.a[1, 0, 0] = 1
    x = torch.zeros(2, 3, 2)
    x[0, 0, 0] = 2
    x[1, 2, 0] = 2
    scores = model.score(x)
    assert torch.allclose(scores[:, 1], torch.tensor([2 / 3, 0.0]))

    with pytest.raises(ValueError, match="locked to the fit-time shape"):
        model.score(torch.zeros(2, 4, 2))


def test_unshared_image_weights_preserve_masks_and_chunking():
    model = RoTImage(2, (1, 2, 3), share_weights=False)
    assert tuple(model.a.shape) == (2, 1, 2, 3)
    with torch.no_grad():
        model.a[1, 0, 0, 0] = 1

    x = torch.zeros(2, 1, 2, 3)
    x[0, 0, 0, 0] = 2
    mask = torch.ones(2, 2, 3, dtype=torch.bool)
    mask[1, 1] = False
    dense = model._unit_importance(x, mask=mask)
    chunked = model._unit_importance(x, mask=mask, sample_chunk=1, class_chunk=1)
    assert torch.allclose(dense, chunked)
    assert torch.all(dense[1, :, 1] == 0)

    with pytest.raises(ValueError, match="locked to the fit-time shape"):
        model.score(torch.zeros(2, 1, 3, 3))


def test_facade_accepts_unshared_weights_and_persists_them(tmp_path):
    rng = np.random.RandomState(0)
    text_x = rng.randn(10, 3, 2).astype(np.float32)
    text_y = (text_x[:, 0, 0] > 0).astype(np.int64)
    text_exp = fit_text(
        text_y,
        text_x,
        share_weights=False,
        epochs=2,
        pretrain_epochs=0,
        batch_size=10,
        learning_rate=0.01,
        l1_penalty=0,
        weight_decay=0,
        seed=0,
    )
    assert tuple(text_exp.model.a.shape) == (2, 3, 2)

    image_x = rng.randn(10, 1, 2, 3).astype(np.float32)
    image_y = (image_x[:, 0, 0, 0] > 0).astype(np.int64)
    image_exp = fit_image(
        image_y,
        image_x,
        share_weights=False,
        epochs=2,
        pretrain_epochs=0,
        batch_size=10,
        learning_rate=0.01,
        weight_decay=0,
        seed=0,
    )
    assert tuple(image_exp.model.a.shape) == (2, 1, 2, 3)

    path = tmp_path / "unshared.rotx"
    image_exp.save(path)
    loaded = load_explainer(path)
    assert loaded.model.share_weights is False
    assert np.allclose(image_exp.get_explanation(image_x), loaded.get_explanation(image_x))
