"""Unit tests for explainer persistence (:meth:`Explainer.save` / :func:`load_explainer`)."""

import numpy as np
import pytest
import torch

from ruleofthumb import fit_image, fit_tabular, fit_text, load_explainer


def _dataset(modality):
    rng = np.random.RandomState(0)
    y = (rng.rand(32) > 0.5).astype(np.int64)
    if modality == "tabular":
        return (rng.rand(32, 4).astype(np.float32), None, y)
    if modality == "text":
        return (rng.rand(32, 6, 4).astype(np.float32), np.array([6, 4] * 16), y)
    return (rng.rand(32, 3, 5, 5).astype(np.float32), None, y)


def _fit(modality, x, padding, y):
    kwargs = {"epochs": 8, "batch_size": 16, "learning_rate": 0.05, "seed": 0}
    if modality == "tabular":
        return fit_tabular(y, x, **kwargs)
    if modality == "text":
        return fit_text(y, x, lengths=padding, **kwargs)
    return fit_image(y, x, **kwargs)


@pytest.mark.parametrize("modality", ["tabular", "text", "image"])
def test_round_trip_preserves_outputs(tmp_path, modality):
    from ruleofthumb.text import lengths_to_mask

    x, padding, y = _dataset(modality)
    exp = _fit(modality, x, padding, y)
    path = tmp_path / "explainer.rotx"
    exp.save(str(path))

    loaded = load_explainer(str(path))
    assert loaded.modality == modality
    assert np.allclose(exp.get_explanation(x, lengths=padding), loaded.get_explanation(x, lengths=padding))
    xt = torch.from_numpy(x)
    mask = lengths_to_mask(padding, x.shape[1]) if padding is not None else None
    assert torch.allclose(exp.predict(xt, mask=mask), loaded.predict(xt, mask=mask))
    assert np.array_equal(exp.get_order(xt, mask=mask), loaded.get_order(xt, mask=mask))


def test_round_trip_restores_mins_maxs(tmp_path):
    x, _, y = _dataset("tabular")
    exp = _fit("tabular", x, None, y)
    exp.save(str(tmp_path / "explainer.rotx"))

    loaded = load_explainer(str(tmp_path / "explainer.rotx"))
    assert torch.allclose(loaded.model.mins, exp.model.mins)
    assert torch.allclose(loaded.model.maxs, exp.model.maxs)


def test_load_with_device_argument(tmp_path):
    x, _, y = _dataset("image")
    exp = _fit("image", x, None, y)
    exp.save(str(tmp_path / "explainer.rotx"))

    loaded = load_explainer(str(tmp_path / "explainer.rotx"), device="cpu")
    assert loaded.model.device.type == "cpu"
    assert np.allclose(exp.get_explanation(x), loaded.get_explanation(x))


def test_foreign_file_rejected(tmp_path):
    path = tmp_path / "foreign.rotx"
    torch.save({"unrelated": 1}, path)
    with pytest.raises(ValueError, match="ruleofthumb explainer"):
        load_explainer(str(path))
