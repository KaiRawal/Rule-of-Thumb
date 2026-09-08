"""Tests pinning v0.2 mask-aware behaviour (replaces the v0.1 limitation pins).

Since v0.2 padding is explicit: masks are passed by callers and no fill value
has special meaning. These tests pin:

- text/image models accept rectangular batches plus validity masks;
- ragged inputs are handled by ``pad_sequences`` / ``pad_images`` utilities;
- the incremental-reveal pipeline (``get_order`` / ``ordered_predict`` /
  ``score_ordering``) is mask-aware;
- legacy ``-1`` sentinel workflows migrate via ``sentinel_mask``.
"""

import numpy as np
import pytest
import torch

from ruleofthumb.core import RoT
from ruleofthumb.image import RoTImage, pad_images
from ruleofthumb.text import RoTText, pad_sequences, sentinel_mask


def lengths_to_bool(lengths, max_len):
    return torch.arange(max_len)[None, :] < lengths[:, None]


@pytest.fixture
def legacy_sentinel_text():
    rng = np.random.RandomState(3)
    x = rng.randn(8, 7, 4).astype(np.float32)
    x[:, -3:, :] = -1.0  # v0.1-style sentinel padding
    lengths = torch.full((8,), 4, dtype=torch.long)
    y = torch.from_numpy((x[:, 0, 0] > 0).astype(np.int64))
    return torch.from_numpy(x), lengths, y


def test_ragged_text_via_pad_sequences():
    seqs = [np.random.RandomState(i).randn(t, 4).astype(np.float32) for i, t in enumerate([3, 7, 5])]
    padded, lengths = pad_sequences(seqs)
    model = RoTText(2, (padded.shape[1], padded.shape[2]))
    imp = model.importance(padded, mask=lengths_to_bool(lengths, padded.shape[1]))
    assert tuple(imp.shape) == (3, 2, 7, 4)
    assert torch.all(imp[0, :, 3:, :] == 0)


def test_v01_sentinel_scores_reproduce_with_explicit_mask(legacy_sentinel_text):
    """Explicit ``sentinel_mask`` + mask-aware scoring matches v0.1 semantics."""
    torch.manual_seed(0)
    x, _, _ = legacy_sentinel_text
    model = RoTText(2, (7, 4))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)

    mask = sentinel_mask(x)
    score = model.score(x, mask=mask).cpu()

    # v0.1 formula: mean over non-padded tokens of per-token importance sums,
    # summed over embedding dims, plus g (computed on CPU with host-side weights).
    a, b, g = model.a.detach().cpu(), model.b.detach().cpu(), model.g.detach().cpu()
    imp = a[None, :, None, :] * (x[:, None] + b[None, :, None, :])
    expected = imp[:, :, :4, :].sum(dim=(2, 3)) / 4 + g[None, :]
    assert torch.allclose(score, expected)
    # without the mask, padded tokens would leak into the score (v0.1 quirk gone)
    assert not torch.allclose(score, model.score(x).cpu())


def test_image_mixed_sizes_fit_and_reveal():
    rng = np.random.RandomState(0)
    imgs = [rng.randn(3, h, w).astype(np.float32) for h, w in [(4, 4), (6, 7), (5, 5)]]
    labels = torch.randint(0, 2, (3,))
    padded, mask = pad_images(imgs)

    model = RoTImage(2, (3,))
    model.fit(padded, labels, epochs=4, batch_size=3, lr=0.05, mask=mask)

    # default granularity: one reveal unit per pixel
    order = model.get_order(padded, mask=mask)
    flat = order.reshape(3, -1)
    true_counts = np.array([h * w for h, w in [(4, 4), (6, 7), (5, 5)]])
    assert (np.asarray(flat != -1).sum(1) == true_counts).all()
    for i in range(3):
        valid = mask[i].reshape(-1).numpy()
        expected_pixels = set(np.flatnonzero(valid).tolist())
        assert sorted(flat[i][: true_counts[i]].tolist()) == sorted(expected_pixels)
        assert (flat[i][true_counts[i] :] == -1).all()


def test_image_element_granularity_escape_hatch():
    rng = np.random.RandomState(0)
    imgs = [rng.randn(3, h, w).astype(np.float32) for h, w in [(4, 4), (6, 7), (5, 5)]]
    labels = torch.randint(0, 2, (3,))
    padded, mask = pad_images(imgs)

    model = RoTImage(2, (3,))
    model.fit(padded, labels, epochs=4, batch_size=3, lr=0.05, mask=mask)

    order = model.get_order(padded, mask=mask, granularity="element")
    flat = np.asarray(order).reshape(3, -1)
    element_counts = np.array([3 * h * w for h, w in [(4, 4), (6, 7), (5, 5)]])
    assert (np.asarray(flat != -1).sum(1) == element_counts).all()
    for i in range(3):
        valid = np.repeat(mask[i].reshape(-1).numpy(), 3)  # pixels -> channel elements
        expected_elements = set(np.flatnonzero(valid).tolist())
        assert sorted(flat[i][: element_counts[i]].tolist()) == sorted(expected_elements)


def test_ordered_predict_truncates_by_default():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 3)
    lengths = torch.tensor([6, 3, 5, 2])
    mask = lengths_to_bool(lengths, 6)
    model = RoTText(2, (6, 3))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)

    order = model.get_order(x, mask=mask)

    # default granularity: one reveal step per token
    pred = model.ordered_predict(x, order)
    assert pred.shape == (4, int(lengths.max()) + 1)
    assert (pred[:, 0] != -1).all()
    for i, length in enumerate(lengths.tolist()):
        assert (pred[i, : length + 1] != -1).all()
        if length + 1 < pred.shape[1]:
            assert (pred[i, length + 1 :] == -1).all()

    full = model.ordered_predict(x, order, include_padded=True)
    assert full.shape == (4, 7)  # all rectangular token steps retained
    assert (full != -1).all()  # legacy-style constant tail retained


def test_text_element_granularity_escape_hatch():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 3)
    lengths = torch.tensor([6, 3, 5, 2])
    mask = lengths_to_bool(lengths, 6)
    model = RoTText(2, (6, 3))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)

    elements = lengths * 3
    order = model.get_order(x, mask=mask, granularity="element")
    assert np.asarray(order).shape == (4, 6, 3)

    pred = model.ordered_predict(x, order, granularity="element")
    assert pred.shape == (4, int(elements.max()) + 1)
    for i, elem in enumerate(elements.tolist()):
        assert (pred[i, : elem + 1] != -1).all()
        if elem + 1 < pred.shape[1]:
            assert (pred[i, elem + 1 :] == -1).all()

    full = model.ordered_predict(x, order, include_padded=True, granularity="element")
    assert full.shape == (4, 19)  # all 6 x 3 rectangular steps retained


def test_unit_curve_matches_element_curve_at_token_boundaries():
    """Fully revealed predictions agree across granularities."""
    torch.manual_seed(1)
    x = torch.randn(4, 6, 3)
    lengths = torch.tensor([6, 4, 5, 2])
    mask = lengths_to_bool(lengths, 6)
    model = RoTText(2, (6, 3))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)

    unit_pred = model.ordered_predict(x, model.get_order(x, mask=mask))
    elem_pred = model.ordered_predict(
        x, model.get_order(x, mask=mask, granularity="element"), granularity="element"
    )
    for i, length in enumerate(lengths.tolist()):
        assert torch.equal(unit_pred[i, length], elem_pred[i, length * 3])


def test_score_ordering_curve_lengths():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 3)
    lengths = torch.tensor([6, 3, 5, 2])
    mask = lengths_to_bool(lengths, 6)
    labels = torch.randint(0, 2, (4,))
    model = RoTText(2, (6, 3))

    order = model.get_order(x, mask=mask)
    metric = model.score_ordering(x, labels, order)
    assert metric.shape == (7,)  # max true tokens + 1

    full = model.score_ordering(x, labels, order, include_padded=True)
    assert full.shape == (7,)
    assert torch.isfinite(full).all()

    elem_metric = model.score_ordering(
        x,
        labels,
        model.get_order(x, mask=mask, granularity="element"),
        granularity="element",
    )
    assert elem_metric.shape == (19,)


def test_unmasked_behaviour_is_unchanged_from_v01():
    """Without masks the ordering pipeline behaves exactly as before."""
    rng = np.random.RandomState(0)
    x = torch.from_numpy(rng.randn(16, 5).astype(np.float32))
    model = RoT(2, (5,))
    model.fit(x, torch.randint(0, 2, (16,)), epochs=4, batch_size=8, lr=0.05)

    order = model.get_order(x)
    for row in order:
        assert sorted(row.tolist()) == list(range(5))  # full permutation, no -1
    assert (order >= 0).all()

    pred = model.ordered_predict(x, order)
    assert pred.shape == (16, 6)
    assert (pred != -1).all()


def test_tabular_both_granularities_identical():
    """Tabular has no sub-feature elements: both settings agree exactly."""
    rng = np.random.RandomState(1)
    x = torch.from_numpy(rng.randn(16, 5).astype(np.float32))
    model = RoT(2, (5,))
    model.fit(x, torch.randint(0, 2, (16,)), epochs=4, batch_size=8, lr=0.05)

    order_unit = model.get_order(x, granularity="unit")
    order_elem = model.get_order(x, granularity="element")
    assert np.array_equal(np.asarray(order_unit), np.asarray(order_elem))

    labels = torch.randint(0, 2, (16,))
    metric_unit = model.score_ordering(x, labels, order_unit)
    metric_elem = model.score_ordering(x, labels, order_elem, granularity="element")
    assert torch.equal(metric_unit, metric_elem)


def test_legacy_imports_are_gone():
    import ruleofthumb

    assert not hasattr(ruleofthumb, "RoT_text")
    assert not hasattr(ruleofthumb, "RoT_image")


@pytest.mark.parametrize("util", ["pad_sequences", "sentinel_mask", "pad_images"])
def test_padding_utils_exported(util):
    import ruleofthumb

    assert hasattr(ruleofthumb, util)
