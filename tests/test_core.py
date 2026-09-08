import numpy as np
import pytest
import torch

from ruleofthumb.core import RoT
from ruleofthumb.image import RoTImage


@pytest.fixture
def tabular_data():
    rng = np.random.RandomState(0)
    x = rng.randn(64, 5).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)  # simple linear rule (CrossEntropy needs int labels)
    return x, torch.from_numpy(y)


def test_fit_reduces_loss_and_predicts(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    model.fit(torch.from_numpy(x), y, epochs=8, batch_size=32, lr=0.05)
    assert model.training_loss is not None
    assert len(model.training_loss) == 8
    assert model.training_loss[-1] <= model.training_loss[0]

    score = model.score(torch.from_numpy(x))
    assert score.shape == (64, 2)
    pred = model.predict(torch.from_numpy(x)).cpu()
    assert pred.shape == (64,)
    accuracy = (pred == y).float().mean().item()
    assert accuracy > 0.7  # the rule is trivially learnable


def test_importance_shape(tabular_data):
    x, _ = tabular_data
    model = RoT(2, (5,))
    imp = model.importance(torch.from_numpy(x))
    assert tuple(imp.shape) == (64, 2, 5)


def test_ordering_pipeline(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    model.fit(torch.from_numpy(x), y, epochs=4, batch_size=32, lr=0.05)

    order = model.get_order(torch.from_numpy(x))
    assert order.shape == (64, 5)
    # each row is a permutation of 0..4
    for row in order:
        assert sorted(row.tolist()) == list(range(5))

    pred = model.ordered_predict(torch.from_numpy(x), order)
    assert pred.shape == (64, 6)

    labels = torch.randint(0, 2, (64,))
    metric = model.score_ordering(torch.from_numpy(x), labels, order)
    # one fidelity value per prefix of revealed features (0..d features shown)
    assert metric.shape == (6,)


def test_score_ordering_multiclass_accuracy_and_confusion():
    rng = np.random.RandomState(2)
    x = torch.from_numpy(rng.randn(24, 5).astype(np.float32))
    y3 = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()  # labels in {0, 1, 2}
    model = RoT(3, (5,))
    model.fit(x, y3, epochs=4, batch_size=8, lr=0.05)

    order = model.get_order(x)
    acc = model.score_ordering(x, y3, order)

    pred = model.ordered_predict(x, order).cpu()
    expected = torch.stack(
        [(pred[:, j] == y3)[pred[:, j] != -1].float().mean() for j in range(pred.shape[1])]
    )
    assert acc.shape == (pred.shape[1],)
    assert torch.allclose(acc, expected)

    confusion = model.score_ordering(x, y3, order, return_confusion=True)
    assert confusion.shape == (pred.shape[1], 3, 3)
    # rows sum to the number of still-active samples at each step
    assert torch.equal(confusion.sum((1, 2)), (pred != -1).float().sum(0))
    # spot-check one step against a manual tally (rows = truth, cols = prediction)
    manual = torch.zeros(3, 3, dtype=torch.long)
    sel = pred[:, 2] != -1
    for t, p in zip(y3[sel].tolist(), pred[sel, 2].tolist()):
        manual[t, p] += 1
    assert torch.equal(confusion[2], manual)


def test_score_ordering_binary_default_matches_count_metric(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    model.fit(torch.from_numpy(x), y, epochs=4, batch_size=32, lr=0.05)
    order = model.get_order(torch.from_numpy(x))
    default = model.score_ordering(torch.from_numpy(x), y, order)
    counts = model.score_ordering(
        torch.from_numpy(x), y, order, metric=lambda tp, fp, fn, tn: (tp + tn) / (tp + fp + fn + tn)
    )
    assert torch.allclose(default, counts)


def test_score_ordering_confusion_conflicts_with_custom_metric(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    order = model.get_order(torch.from_numpy(x))
    with pytest.raises(ValueError):
        model.score_ordering(
            torch.from_numpy(x), y, order, metric=lambda tp, fp, fn, tn: tp, return_confusion=True
        )


def test_device_resolution_defaults_to_available_backend():
    from ruleofthumb.core import _resolve_device

    assert _resolve_device("cpu") == torch.device("cpu")
    dev = _resolve_device(None)
    assert dev.type in {"cpu", "cuda", "mps"}


def test_device_parameter_cpu_plumbing(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,), device="cpu")
    assert model.a.device.type == "cpu"
    assert model.g.device.type == "cpu"

    model.fit(x, y, epochs=4, batch_size=32, lr=0.05)
    order = model.get_order(x)
    pred = model.ordered_predict(torch.from_numpy(x), order)
    assert pred.device.type == "cpu"
    metric = model.score_ordering(torch.from_numpy(x), y, order)
    assert metric.shape == (6,)

    image_model = RoTImage(2, (3,), device="cpu")
    assert image_model.a.device.type == "cpu"


def test_stochastic_importance_dropout():
    torch.manual_seed(0)
    x = torch.randn(16, 3)
    model = RoT(2, (3,), dropout_rate=0.5)
    imp = model.stochastic_importance(x)
    assert tuple(imp.shape) == (16, 2, 3)


def test_mins_maxs_are_instance_attributes():
    model_a = RoT(2, (3,))
    model_b = RoT(2, (3,))
    assert model_a.mins == -np.inf
    assert model_a.maxs == np.inf

    # mutating one instance must not leak into others or the class
    model_a.fit_project(-1.0, 1.0)
    assert model_b.mins == -np.inf
    assert model_b.maxs == np.inf
    assert not hasattr(RoT, "mins")
    assert not hasattr(RoT, "maxs")


def test_get_order_stable_tie_breaking(tabular_data):
    x, _ = tabular_data
    model = RoT(2, (5,))
    # unfitted model: a and b are zero, so every feature has equal importance
    order = model.get_order(torch.from_numpy(x))
    expected = np.arange(5)
    for row in order:
        assert np.array_equal(row, expected)

    # repeated calls are deterministic
    again = model.get_order(torch.from_numpy(x))
    assert np.array_equal(order, again)


def test_training_loop_swa_burn_in_parameter(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.05)

    # default burn-in (epochs // 10 + 1 == 1) creates the SWA model
    model.training_loop(model.loss, torch.from_numpy(x), y, optimiser, epochs=2, batch_size=32)
    assert model.swa_model is not None

    # a burn-in beyond the epoch count never creates it
    model = RoT(2, (5,))
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.05)
    model.training_loop(model.loss, torch.from_numpy(x), y, optimiser, epochs=2, batch_size=32, swa_burn_in=5)
    assert model.swa_model is None

    # an explicit zero burn-in works too
    model = RoT(2, (5,))
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.05)
    model.training_loop(model.loss, torch.from_numpy(x), y, optimiser, epochs=2, batch_size=32, swa_burn_in=0)
    assert model.swa_model is not None


def test_fit_hyperparameter_arguments(tabular_data):
    x, y = tabular_data
    model = RoT(2, (5,))
    model.fit(
        torch.from_numpy(x), y, epochs=8, batch_size=32, lr=0.05, pretrain_epochs=2, weight_decay=0.1
    )
    assert len(model.training_loss) == 8
    assert model.training_loss[-1] <= model.training_loss[0]


def test_fit_seed_reproducibility(tabular_data):
    x, y = tabular_data
    x = torch.from_numpy(x)

    model_a = RoT(2, (5,))
    model_a.fit(x, y, epochs=8, batch_size=32, lr=0.05, seed=42)
    model_b = RoT(2, (5,))
    model_b.fit(x, y, epochs=8, batch_size=32, lr=0.05, seed=42)
    assert np.array_equal(model_a.training_loss, model_b.training_loss)
    assert torch.equal(model_a.score(x), model_b.score(x))

    model_c = RoT(2, (5,))
    model_c.fit(x, y, epochs=8, batch_size=32, lr=0.05, seed=7)
    assert not np.array_equal(model_a.training_loss, model_c.training_loss)


def test_training_loop_seed_reproducibility(tabular_data):
    x, y = tabular_data
    x = torch.from_numpy(x)

    losses = []
    for _ in range(2):
        model = RoT(2, (5,), dropout_rate=0.5)
        optimiser = torch.optim.AdamW(model.parameters(), lr=0.05)
        model.training_loop(model.loss, x, y, optimiser, epochs=4, batch_size=32, seed=123)
        losses.append(model.training_loss)
    assert np.array_equal(losses[0], losses[1])
