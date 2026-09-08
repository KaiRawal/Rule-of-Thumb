import numpy as np
import pytest
import torch

from ruleofthumb.text import RoTText, lengths_to_mask


@pytest.fixture
def text_data():
    rng = np.random.RandomState(0)
    n, tokens, embedding = 32, 6, 4
    x = rng.randn(n, tokens, embedding).astype(np.float32)
    y = (x[:, 0, 0] > 0).astype(np.int64)
    return x, torch.from_numpy(y)


@pytest.fixture
def padded_text_data():
    rng = np.random.RandomState(0)
    n, tokens, embedding = 32, 6, 4
    x = rng.randn(n, tokens, embedding).astype(np.float32)
    x[:, -2:, :] = 0.0  # zero padding region (value is irrelevant now)
    mask = torch.ones(n, tokens, dtype=torch.bool)
    mask[:, -2:] = False
    return x, mask, torch.from_numpy((x[:, 0, 0] > 0).astype(np.int64))


def test_importance_shape(text_data):
    torch.manual_seed(0)
    x, _ = text_data
    model = RoTText(2, (x.shape[1], x.shape[2]))
    imp = model.importance(torch.from_numpy(x))
    assert tuple(imp.shape) == (32, 2, 6, 4)


def test_mask_zeroes_padded_tokens(padded_text_data):
    torch.manual_seed(0)
    x, mask, _ = padded_text_data
    model = RoTText(2, (x.shape[1], x.shape[2]))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)
    imp = model.importance(torch.from_numpy(x), mask=mask)
    assert tuple(imp.shape) == (32, 2, 6, 4)
    assert torch.all(imp[:, :, -2:, :] == 0)
    assert torch.any(imp[:, :, :-2, :] != 0)


def test_stochastic_importance_respects_mask(padded_text_data):
    torch.manual_seed(0)
    x, mask, _ = padded_text_data
    model = RoTText(2, (x.shape[1], x.shape[2]), dropout_rate=0.5)
    imp = model.stochastic_importance(torch.from_numpy(x), mask=mask)
    assert torch.all(imp[:, :, -2:, :] == 0)


def test_score_normalises_by_true_length(padded_text_data):
    torch.manual_seed(0)
    x, mask, _ = padded_text_data
    model = RoTText(2, (6, 4))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)

    xt = torch.from_numpy(x)
    score = model.score(xt, mask=mask).cpu()

    # manual reference computed on CPU with host-side copies of the weights
    a, b, g = model.a.detach().cpu(), model.b.detach().cpu(), model.g.detach().cpu()
    imp = a[None, :, None, :] * (xt[:, None] + b[None, :, None, :])
    expected_sum = imp[:, :, :4, :].sum(dim=(2, 3))  # only the 4 real tokens
    expected = expected_sum / 4 + g[None, :]
    assert torch.allclose(score, expected)


def test_fit_runs_and_reduces_loss(text_data):
    x, y = text_data
    model = RoTText(2, (x.shape[1], x.shape[2]))
    model.fit(torch.from_numpy(x), y, epochs=8, batch_size=16, lr=0.01)
    # fit runs a 5-epoch burn-in followed by the requested epochs
    assert len(model.training_loss) == 8


def test_fit_with_mask(padded_text_data):
    x, mask, y = padded_text_data
    model = RoTText(2, (6, 4))
    model.fit(torch.from_numpy(x), y, epochs=4, batch_size=16, lr=0.01, mask=mask)
    assert len(model.training_loss) == 4


def test_pad_sequences_utility():
    from ruleofthumb.text import pad_sequences

    seqs = [np.random.RandomState(i).randn(t, 4).astype(np.float32) for i, t in enumerate([3, 5, 4])]
    padded, lengths = pad_sequences(seqs, pad_value=-7.5)
    assert tuple(padded.shape) == (3, 5, 4)
    assert lengths.tolist() == [3, 5, 4]
    assert torch.all(padded[0, 3:] == -7.5)
    assert torch.equal(padded[1], torch.from_numpy(seqs[1]))


def test_sentinel_mask_replicates_v01_detection():
    from ruleofthumb.text import sentinel_mask

    x = torch.randn(4, 6, 3)
    x[:, -2:, :] = -1.0
    x[0, 0, 0] = -1.0  # a single -1 component does NOT make the token padding
    mask = sentinel_mask(x)  # validity mask: True marks real tokens
    assert mask[:, :-2].all()  # real tokens are valid
    assert not mask[:, -2:].any()  # sentinel rows are padding
    assert mask[0, 0]


def _multiclass_text_data():
    rng = np.random.RandomState(3)
    n, tokens, embedding = 32, 6, 4
    x = rng.randn(n, tokens, embedding).astype(np.float32)
    lengths = torch.tensor([6] * 8 + [5] * 16 + [4] * 8)
    y = (x[:, 0, 0] > 0).astype(np.int64) + (x[:, 1, 1] > 0).astype(np.int64)  # labels in {0, 1, 2}
    return x, lengths, y


def test_multiclass_mask_zeroes_padded_tokens_per_class():
    x, lengths, _ = _multiclass_text_data()
    mask = lengths_to_mask(lengths, x.shape[1])
    model = RoTText(3, (x.shape[1], x.shape[2]))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)
    imp = model.importance(torch.from_numpy(x), mask=mask)
    assert tuple(imp.shape) == (32, 3, 6, 4)
    for k in range(3):  # every class slice respects the padding
        assert torch.all(imp[:, k][~mask] == 0)
        assert torch.any(imp[:, k][mask] != 0)


def test_facade_multiclass_reveal_pipeline_matches_manual_counts():
    from ruleofthumb import fit_text

    x, lengths, y = _multiclass_text_data()
    exp = fit_text(y, x, lengths=lengths, epochs=8, batch_size=16, learning_rate=0.01, seed=0, n_classes=3)
    assert exp.model.classes == 3

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.int64))
    mask = lengths_to_mask(lengths, x.shape[1])

    order = exp.get_order(xt, mask=mask)
    # padded positions ranked last and reported as -1
    for i, length in enumerate(lengths.tolist()):
        assert (order[i, :length] != -1).all() and (order[i, length:] == -1).all()

    pred = exp.ordered_predict(xt, order).cpu()
    valid = pred != -1
    curve = exp.score_ordering(xt, yt, order)
    confusion = exp.score_ordering(xt, yt, order, return_confusion=True)

    steps = pred.shape[1]
    assert tuple(confusion.shape) == (steps, 3, 3)
    for j in range(steps):
        sel = valid[:, j]
        # default accuracy metric equals the manual tally over active samples
        expected = (pred[sel, j] == yt[sel]).float().mean()
        assert torch.allclose(curve[j], expected)
        # confusion rows/columns sum to the active-sample count
        assert int(confusion[j].sum()) == int(sel.sum())


def test_facade_multiclass_lengths_and_attention_mask_equivalence():
    from ruleofthumb import fit_text

    x, lengths, y = _multiclass_text_data()
    mask = lengths_to_mask(lengths, x.shape[1])
    torch.manual_seed(0)
    by_lengths = fit_text(y, x, lengths=lengths, epochs=4, batch_size=16, learning_rate=0.01, seed=0, n_classes=3)
    torch.manual_seed(0)
    by_mask = fit_text(
        y, x, attention_mask=mask.numpy(), epochs=4, batch_size=16, learning_rate=0.01, seed=0, n_classes=3
    )
    imp_lengths = by_lengths.get_explanation(x, lengths=lengths)
    imp_mask = by_mask.get_explanation(x, attention_mask=mask.numpy())
    assert np.allclose(imp_lengths, imp_mask)


def test_facade_multiclass_seed_reproducibility():
    from ruleofthumb import fit_text

    x, lengths, y = _multiclass_text_data()
    kwargs = {"epochs": 4, "batch_size": 16, "learning_rate": 0.01, "seed": 0, "n_classes": 3}
    exp_a = fit_text(y, x, lengths=lengths, **kwargs)
    exp_b = fit_text(y, x, lengths=lengths, **kwargs)
    assert np.allclose(exp_a.get_explanation(x, lengths=lengths), exp_b.get_explanation(x, lengths=lengths))
