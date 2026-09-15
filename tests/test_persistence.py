"""Unit tests for explainer persistence (:meth:`Explainer.save` / :func:`load_explainer`).

Fitted models arrive as committed mock weights (``tests/_mock_weights.py``):
these tests assert on save/load round-trips, so the fit itself is incidental.
The ``.rotx`` path stays fully exercised — mock weights arrive as raw
``.pt`` state dicts, never ``.rotx``.
"""

import numpy as np
import pytest
import torch
from _mock_weights import data_persist, load_mock

from ruleofthumb import load_explainer


def _fit(modality, x, padding, y):
    """Load the committed mock fit (datasets match the builders by construction)."""
    del x, padding, y
    return load_mock(f"persist_{modality}")


def _dataset(modality):
    return data_persist(modality)


@pytest.mark.parametrize("modality", ["tabular", "text", "image"])
def test_round_trip_preserves_outputs(tmp_path, modality):
    x, padding, y = _dataset(modality)
    exp = _fit(modality, x, padding, y)
    path = tmp_path / "explainer.rotx"
    exp.save(str(path))

    loaded = load_explainer(str(path))
    assert loaded.modality == modality
    assert np.allclose(exp.get_explanation(x, mask=padding), loaded.get_explanation(x, mask=padding))
    xt = torch.from_numpy(x)
    mask = None if padding is None else torch.from_numpy(padding)
    assert torch.allclose(exp.predict(xt, mask=mask), loaded.predict(xt, mask=mask))
    assert np.array_equal(exp.get_order(xt, mask=mask), loaded.get_order(xt, mask=mask))


def test_round_trip_restores_mins_maxs(tmp_path):
    x, _, y = _dataset("tabular")
    exp = _fit("tabular", x, None, y)
    exp.save(str(tmp_path / "explainer.rotx"))

    loaded = load_explainer(str(tmp_path / "explainer.rotx"), device="cpu")
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


def test_version_mismatch_rejected(tmp_path, monkeypatch):
    import ruleofthumb

    x, _, y = _dataset("tabular")
    exp = _fit("tabular", x, None, y)
    path = tmp_path / "explainer.rotx"
    monkeypatch.setattr(ruleofthumb, "__version__", "9.9.9")
    exp.save(str(path))
    monkeypatch.undo()
    with pytest.raises(ValueError, match="9.9.9"):
        load_explainer(str(path))


def test_missing_version_rejected(tmp_path):
    x, _, y = _dataset("tabular")
    exp = _fit("tabular", x, None, y)
    path = tmp_path / "explainer.rotx"
    exp.save(str(path))
    payload = torch.load(path, weights_only=True)
    del payload["ruleofthumb_version"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="version"):
        load_explainer(str(path))
