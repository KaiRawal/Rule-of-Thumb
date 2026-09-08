import numpy as np
import pytest
import torch


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
    import ruleofthumb

    x_tab, y_tab = tabular_data
    tx, lengths, ty = text_data
    ix, _imask, iy = image_data

    exp = ruleofthumb.fit(y_tab, x_tab)
    assert exp.modality == "tabular"
    exp = ruleofthumb.fit(ty, tx, lengths=lengths)
    assert exp.modality == "text"
    ix, imask, iy = image_data
    exp = ruleofthumb.fit(iy, ix, mask=imask.numpy())
    assert exp.modality == "image"


def test_fit_explicit_modality_override(tabular_data):
    import ruleofthumb

    x, y = tabular_data
    exp = ruleofthumb.fit(y, x, modality="tabular")
    assert exp.modality == "tabular"
    with pytest.raises(ValueError, match="unknown modality"):
        ruleofthumb.fit(y, x, modality="video")
    with pytest.raises(ValueError, match="ndim"):
        ruleofthumb.fit(y, x.reshape(-1), modality="auto")


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
    exp = fit_text(y, x, epochs=4, batch_size=16, learning_rate=0.01, lengths=lengths)
    imp = exp.get_explanation(x, lengths=lengths)
    assert imp.shape == (32, 6)


def test_text_attention_mask_matches_lengths(text_data):
    from ruleofthumb import fit_text

    x, lengths, y = text_data
    mask = torch.arange(6)[None, :] < lengths[:, None]
    torch.manual_seed(0)
    by_lengths = fit_text(y, x, epochs=2, batch_size=16, learning_rate=0.01, lengths=lengths)
    torch.manual_seed(0)
    by_mask = fit_text(y, x, epochs=2, batch_size=16, learning_rate=0.01, attention_mask=mask.numpy())
    imp_by_lengths = by_lengths.get_explanation(x, lengths=lengths)
    imp_by_mask = by_mask.get_explanation(x, attention_mask=mask.numpy())
    assert np.allclose(imp_by_lengths, imp_by_mask)


def test_text_multiclass_per_class_output(text_data):
    from ruleofthumb import fit_text

    x, lengths, _ = text_data
    y3 = (x[:, 0, 0] > 0).astype(np.int64) + (x[:, 1, 1] > 0).astype(np.int64)  # labels in {0, 1, 2}
    exp = fit_text(y3, x, epochs=4, batch_size=16, learning_rate=0.01, lengths=lengths, n_classes=3)
    assert exp.model.classes == 3
    imp = exp.get_explanation(x, lengths=lengths)
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

    tab = fit_tabular(y, x, epochs=2, batch_size=32)
    with pytest.raises(ValueError, match="no padding"):
        tab.get_explanation(x, lengths=lengths[:32])

    txt = fit_text(ty, tx, epochs=2, batch_size=16, lengths=lengths)
    with pytest.raises(ValueError, match="at most one"):
        txt.get_explanation(tx, lengths=lengths, attention_mask=np.ones((32, 6), dtype=bool))
    # a plain (N, T) boolean mask is a valid alias for attention_mask on text
    imp_by_alias = txt.get_explanation(tx, mask=(torch.arange(6)[None, :] < lengths[:, None]).numpy())
    assert np.allclose(imp_by_alias, txt.get_explanation(tx, lengths=lengths))

    img = fit_image(iy, ix, epochs=2, batch_size=16)
    with pytest.raises(ValueError, match="mask= only"):
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
    exp_text = fit_text(
        ty,
        tx,
        epochs=4,
        batch_size=16,
        learning_rate=0.01,
        lengths=lengths,
        pretrain_epochs=1,
        weight_decay=0.1,
        l1_penalty=0.05,
    )
    assert exp_text.get_explanation(tx, lengths=lengths).shape == (32, 6)


def test_explainer_seed_reproducibility(text_data):
    from ruleofthumb import fit_text

    x, lengths, y = text_data
    exp_a = fit_text(y, x, epochs=4, batch_size=16, learning_rate=0.01, lengths=lengths, seed=0)
    exp_b = fit_text(y, x, epochs=4, batch_size=16, learning_rate=0.01, lengths=lengths, seed=0)
    assert np.allclose(exp_a.get_explanation(x, lengths=lengths), exp_b.get_explanation(x, lengths=lengths))


def test_explainer_device_parameter(tabular_data, text_data):
    from ruleofthumb import fit_tabular, fit_text

    x, y = tabular_data
    exp = fit_tabular(y, x, epochs=4, batch_size=32, learning_rate=0.05, device="cpu")
    assert exp.model.a.device.type == "cpu"
    assert exp.get_explanation(x).shape == (64, 5)

    tx, lengths, ty = text_data
    exp_text = fit_text(ty, tx, epochs=4, batch_size=16, learning_rate=0.01, lengths=lengths, device="cpu")
    assert exp_text.get_explanation(tx, lengths=lengths).shape == (32, 6)


def test_explainer_delegates_reveal_pipeline(tabular_data):
    import ruleofthumb

    x, y = tabular_data
    exp = ruleofthumb.fit(y, x, epochs=4, batch_size=32, learning_rate=0.05)
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
    import ruleofthumb

    assert ruleofthumb.__version__ == "0.0.1"
    for name in ("Explainer", "fit", "fit_tabular", "fit_text", "fit_image", "RoT"):
        assert hasattr(ruleofthumb, name)
    for removed in ("RuleOfThumb", "TextRuleOfThumb"):
        assert not hasattr(ruleofthumb, removed)
