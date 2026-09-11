import numpy as np
import pytest
import torch

from ruleofthumb.text import lengths_to_mask


@pytest.fixture
def tabular_data():
    rng = np.random.RandomState(0)
    x = rng.randn(64, 5).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)  # black-box labels (CrossEntropy needs int)
    return x, y


@pytest.fixture
def text_data():
    rng = np.random.RandomState(1)
    x = rng.randn(32, 6, 4).astype(np.float32)
    lengths = torch.tensor([6] * 8 + [4] * 12 + [5] * 12)
    y = (x[:, 0, 0] > 0).astype(np.int64)
    return x, lengths, y


@pytest.fixture
def image_data():
    rng = np.random.RandomState(2)
    x = rng.randn(16, 3, 6, 6).astype(np.float32)
    mask = torch.ones(16, 6, 6, dtype=torch.bool)
    mask[:, -2:, :] = False  # bottom rows padded
    y = (x[:, :, :, :3].mean(axis=(1, 2, 3)) > 0).astype(np.int64)
    return x, mask, y


def test_fit_auto_detects_modality(tabular_data, text_data, image_data):
    import ruleofthumb as rot

    x_tab, y_tab = tabular_data
    tx, lengths, ty = text_data
    ix, _imask, iy = image_data

    exp = rot.fit(y_tab, x_tab)
    assert exp.modality == "tabular"
    exp = rot.fit(ty, tx, mask=lengths_to_mask(lengths, tx.shape[1]).numpy())
    assert exp.modality == "text"
    ix, imask, iy = image_data
    exp = rot.fit(iy, ix, mask=imask.numpy())
    assert exp.modality == "image"


def test_fit_explicit_modality_override(tabular_data):
    import ruleofthumb as rot

    x, y = tabular_data
    exp = rot.fit(y, x, modality="tabular")
    assert exp.modality == "tabular"
    with pytest.raises(ValueError, match="unknown modality"):
        rot.fit(y, x, modality="video")
    with pytest.raises(ValueError, match="ndim"):
        rot.fit(y, x.reshape(-1), modality="auto")


def test_explainer_explanation_shape(tabular_data):
    from ruleofthumb import fit_tabular

    x, y = tabular_data
    exp = fit_tabular(y, x, epochs=4, batch_size=32, learning_rate=0.05)
    assert exp.get_explanation(x).shape == (64, 5)


def test_explainer_signed_explanations(tabular_data):
    from ruleofthumb import fit_tabular

    x, y = tabular_data
    exp = fit_tabular(y, x, epochs=8, batch_size=32, learning_rate=0.05, seed=0)
    imp = exp.get_explanation(x)

    # signed: both positive and negative contributions exist
    assert (imp > 0).any()
    assert (imp < 0).any()

    # additive decomposition: sum of contributions + class-1 bias = score
    score1 = exp.model.score(torch.from_numpy(x))[:, 1].detach().cpu().numpy()
    bias1 = exp.model.g[1].detach().cpu().numpy()
    assert np.allclose(imp.sum(1) + bias1, score1, atol=1e-4)


def test_explainer_multiclass_per_class_output(tabular_data):
    from ruleofthumb import fit_tabular

    x, _ = tabular_data
    y3 = (x[:, 0] > 0).astype(np.int64) + (x[:, 1] > 0).astype(np.int64)  # labels in {0, 1, 2}
    exp = fit_tabular(y3, x, epochs=4, batch_size=32, learning_rate=0.05, n_classes=3)
    imp = exp.get_explanation(x)
    assert imp.shape == (64, 3, 5)
    for k in range(3):
        assert (imp[:, k, :] != 0).any()


def test_text_explanation_shape(text_data):
    from ruleofthumb import fit_text

    x, lengths, y = text_data
    mask = lengths_to_mask(lengths, x.shape[1]).numpy()
    exp = fit_text(y, x, epochs=4, batch_size=16, learning_rate=0.01, mask=mask)
    imp = exp.get_explanation(x, mask=mask)
    assert imp.shape == (32, 6)


def test_text_mask_from_lengths_matches_manual(text_data):
    """The documented ``lengths_to_mask`` bridge equals a hand-built mask."""
    from ruleofthumb import fit_text

    x, lengths, y = text_data
    mask = lengths_to_mask(lengths, x.shape[1]).numpy()
    manual = (torch.arange(6)[None, :] < lengths[:, None]).numpy()
    assert (mask == manual).all()
    torch.manual_seed(0)
    by_bridge = fit_text(y, x, epochs=2, batch_size=16, learning_rate=0.01, mask=mask)
    torch.manual_seed(0)
    by_manual = fit_text(y, x, epochs=2, batch_size=16, learning_rate=0.01, mask=manual)
    assert np.allclose(
        by_bridge.get_explanation(x, mask=mask), by_manual.get_explanation(x, mask=manual)
    )


def test_text_multiclass_per_class_output(text_data):
    from ruleofthumb import fit_text

    x, lengths, _ = text_data
    y3 = (x[:, 0, 0] > 0).astype(np.int64) + (x[:, 1, 1] > 0).astype(np.int64)  # labels in {0, 1, 2}
    mask = lengths_to_mask(lengths, x.shape[1]).numpy()
    exp = fit_text(y3, x, epochs=4, batch_size=16, learning_rate=0.01, mask=mask, n_classes=3)
    assert exp.model.classes == 3
    imp = exp.get_explanation(x, mask=mask)
    # K > 2: full per-class output, class axis not collapsed
    assert imp.shape == (32, 3, 6)
    for k in range(3):
        assert (imp[:, k, :] != 0).any()
        assert (imp[:, k, :] > 0).any() and (imp[:, k, :] < 0).any()


def test_image_explanation_shape_and_padding(image_data):
    from ruleofthumb import fit_image

    x, mask, y = image_data
    exp = fit_image(y, x, mask=mask.numpy(), epochs=4, batch_size=16, learning_rate=0.05)
    imp = exp.get_explanation(x, mask=mask.numpy())
    assert exp.modality == "image"
    assert imp.shape == (16, 6, 6)
    # padded pixels score exactly zero
    assert (imp[:, -2:, :] == 0).all()
    assert (imp != 0).any()


def test_image_multiclass_per_class_output(image_data):
    from ruleofthumb import fit_image

    x, mask, _ = image_data
    y3 = (x[:, 0, 0, 0] > 0).astype(np.int64) + (x[:, 0, 1, 1] > 0).astype(np.int64)  # {0, 1, 2}
    exp = fit_image(y3, x, mask=mask.numpy(), epochs=4, batch_size=16, learning_rate=0.05, n_classes=3)
    imp = exp.get_explanation(x, mask=mask.numpy())
    assert imp.shape == (16, 3, 6, 6)
    for k in range(3):
        assert (imp[:, k, -2:, :] == 0).all()  # padding respected per class
        assert (imp[:, k, :-2, :] != 0).any()


def test_modality_specific_padding_arguments_rejected(tabular_data, text_data, image_data):
    from ruleofthumb import fit_image, fit_tabular, fit_text

    x, y = tabular_data
    tx, lengths, ty = text_data
    ix, _imask, iy = image_data
    tmask = lengths_to_mask(lengths, tx.shape[1]).numpy()

    tab = fit_tabular(y, x, epochs=2, batch_size=32)
    with pytest.raises(ValueError, match="no mask"):
        tab.get_explanation(x, mask=np.ones((64, 5), dtype=bool))

    # text speaks mask= only: the retired lengths=/attention_mask= spellings are TypeErrors
    txt = fit_text(ty, tx, epochs=2, batch_size=16, mask=tmask)
    with pytest.raises(TypeError):
        txt.get_explanation(tx, lengths=lengths)
    order = txt.get_order(torch.from_numpy(tx), mask=tmask)
    assert (order[:, -1] == -1).any()  # padding ranked last

    img = fit_image(iy, ix, epochs=2, batch_size=16)
    with pytest.raises(TypeError):
        img.get_explanation(ix, lengths=torch.tensor([6] * 16))


def test_factories_reject_foreign_kwargs(tabular_data, image_data):
    from ruleofthumb import fit, fit_image, fit_tabular

    x, y = tabular_data
    ix, _imask, iy = image_data
    with pytest.raises(TypeError, match="l1_penalty"):
        fit_tabular(y, x, l1_penalty=0.01)
    with pytest.raises(TypeError, match="lengths"):
        fit(y, x, modality="tabular", lengths=torch.tensor([5] * 64))
    with pytest.raises(TypeError, match="attention_mask"):
        fit_image(iy, ix, attention_mask=np.ones((16, 6, 6), dtype=bool))


def test_explainer_threads_training_hyperparameters(tabular_data, text_data):
    from ruleofthumb import fit_tabular, fit_text

    x, y = tabular_data
    exp = fit_tabular(y, x, epochs=4, batch_size=32, learning_rate=0.05, pretrain_epochs=1, weight_decay=0.1)
    assert exp.get_explanation(x).shape == (64, 5)

    tx, lengths, ty = text_data
    tmask = lengths_to_mask(lengths, tx.shape[1]).numpy()
    exp_text = fit_text(
        ty,
        tx,
        epochs=4,
        batch_size=16,
        learning_rate=0.01,
        mask=tmask,
        pretrain_epochs=1,
        weight_decay=0.1,
        l1_penalty=0.05,
    )
    assert exp_text.get_explanation(tx, mask=tmask).shape == (32, 6)


def test_explainer_seed_reproducibility(text_data):
    from ruleofthumb import fit_text

    x, lengths, y = text_data
    mask = lengths_to_mask(lengths, x.shape[1]).numpy()
    exp_a = fit_text(y, x, epochs=4, batch_size=16, learning_rate=0.01, mask=mask, seed=0)
    exp_b = fit_text(y, x, epochs=4, batch_size=16, learning_rate=0.01, mask=mask, seed=0)
    assert np.allclose(exp_a.get_explanation(x, mask=mask), exp_b.get_explanation(x, mask=mask))


def test_explainer_device_parameter(tabular_data, text_data):
    from ruleofthumb import fit_tabular, fit_text

    x, y = tabular_data
    exp = fit_tabular(y, x, epochs=4, batch_size=32, learning_rate=0.05, device="cpu")
    assert exp.model.a.device.type == "cpu"
    assert exp.get_explanation(x).shape == (64, 5)

    tx, lengths, ty = text_data
    tmask = lengths_to_mask(lengths, tx.shape[1]).numpy()
    exp_text = fit_text(ty, tx, epochs=4, batch_size=16, learning_rate=0.01, mask=tmask, device="cpu")
    assert exp_text.get_explanation(tx, mask=tmask).shape == (32, 6)


def test_explainer_delegates_reveal_pipeline(tabular_data):
    import ruleofthumb as rot

    x, y = tabular_data
    exp = rot.fit(y, x, epochs=4, batch_size=32, learning_rate=0.05)
    order = exp.get_order(torch.from_numpy(x))
    assert order.shape == (64, 5)

    pred = exp.ordered_predict(torch.from_numpy(x), order)
    assert pred.shape == (64, 6)

    curve = exp.score_ordering(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)), order)
    assert curve.shape == (6,)

    scores = exp.score(torch.from_numpy(x))
    assert scores.shape == (64, 2)
    preds = exp.predict(torch.from_numpy(x))
    assert preds.shape == (64,)


def test_package_exports():
    import ruleofthumb as rot

    assert rot.__version__ == "0.0.2"
    for name in ("Explainer", "fit", "fit_tabular", "fit_text", "fit_image", "RoT"):
        assert hasattr(rot, name)
    for removed in ("RuleOfThumb", "TextRuleOfThumb"):
        assert not hasattr(rot, removed)


def _mps_available():
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


@pytest.mark.skipif(not _mps_available(), reason="MPS device not available")
def test_inference_returns_host_side_on_mps(tabular_data):
    """Sandbox Bug 8 repro: predict/score must survive np.asarray(...) on MPS."""
    import ruleofthumb as rot

    x, y = tabular_data
    exp = rot.fit(y, x, modality="tabular", epochs=4, batch_size=32, learning_rate=0.05, seed=0, device="mps")
    assert exp.model.a.device.type == "mps"  # training really ran on MPS

    preds = np.asarray(exp.predict(torch.from_numpy(x)))
    assert preds.shape == (x.shape[0],)
    assert exp.score(torch.from_numpy(x)).device.type == "cpu"
    order = exp.get_order(torch.from_numpy(x))
    assert exp.ordered_predict(torch.from_numpy(x), order).device.type == "cpu"


def test_factories_reject_out_of_range_labels():
    """A 1000-class head against n_classes=2 must fail clearly, not in torch (Bug 1 follow-up)."""
    from ruleofthumb import fit_image, fit_tabular, fit_text

    rng = np.random.RandomState(0)
    y_bad = np.array([0, 1, 549, 2, 0, 1])
    with pytest.raises(ValueError, match="n_classes"):
        fit_tabular(y_bad, rng.rand(6, 3).astype(np.float32), epochs=1, batch_size=6)
    with pytest.raises(ValueError, match="n_classes"):
        fit_text(y_bad, rng.rand(6, 4, 3).astype(np.float32), epochs=1, batch_size=6)
    with pytest.raises(ValueError, match="n_classes"):
        fit_image(y_bad, rng.rand(6, 2, 4, 4).astype(np.float32), epochs=1, batch_size=6)
    with pytest.raises(ValueError, match="n_classes"):
        fit_tabular(np.array([0, -1]), rng.rand(2, 3).astype(np.float32), epochs=1, batch_size=2)
    with pytest.raises(ValueError, match="non-empty"):
        fit_tabular(np.array([], dtype=np.int64), rng.rand(0, 3).astype(np.float32), epochs=1, batch_size=6)


def test_readonly_arrays_convert_silently(tmp_path):
    """Bug 6 extension: mmap/read-only inputs must not warn (copy before convert)."""
    import warnings

    from ruleofthumb import fit_tabular

    rng = np.random.RandomState(0)
    x = rng.rand(32, 4).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    path = tmp_path / "x.npy"
    np.save(path, x)
    xr = np.load(path, mmap_mode="r")
    assert not xr.flags.writeable
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exp = fit_tabular(y, xr, epochs=2, batch_size=32, learning_rate=0.05, seed=0)
        imp = exp.get_explanation(xr)
    assert imp.shape == (32, 4)


def test_oversized_batch_warns(tabular_data):
    """Bug 9: an oversized batch warns loudly instead of silently running full-batch."""
    from ruleofthumb import fit_tabular

    x, y = tabular_data
    with pytest.warns(UserWarning, match="batch_size"):
        exp = fit_tabular(y, x, epochs=2, batch_size=5000, learning_rate=0.05, seed=0)
    assert len(exp.model.training_loss) == 2


def test_fit_quality_warns_on_random_labels():
    """Bug 10: near-chance train agreement warns and is inspectable."""
    from ruleofthumb import fit_tabular

    rng = np.random.RandomState(0)
    x = rng.rand(128, 4).astype(np.float32)
    y = rng.randint(0, 2, 128).astype(np.int64)
    with pytest.warns(UserWarning, match="train agreement"):
        exp = fit_tabular(y, x, epochs=4, batch_size=32, learning_rate=0.05, seed=0)
    assert exp.train_agreement_ < 0.75


def test_fit_quality_quiet_on_clean_task(tabular_data):
    """Bug 10: a clean fit records high agreement with no warning."""
    import warnings

    from ruleofthumb import fit_tabular

    x, y = tabular_data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exp = fit_tabular(y, x, epochs=8, batch_size=32, learning_rate=0.05, seed=0)
    assert exp.train_agreement_ >= 0.75
