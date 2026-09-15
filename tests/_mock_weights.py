"""Mock fitted RoT weights for unit tests (ToDo item 22, unit-mock direction).

Unit tests that assert on the *outputs* of a pre-fit RoT model (explanations,
plots, persistence round-trips, mechanism recovery) load committed raw
``.pt`` state dicts instead of fitting live. Raw state dicts only — never
``.rotx`` (``load_explainer`` enforces an exact package-version match, which
would break these fixtures on every release; the ``.rotx`` path is covered
by the round-trip tests themselves).

Dataset builders live here (single-sourced): both
``tests/mint_unit_weights.py`` and the unit tests import them, so the mint
fit and the test inputs can never drift apart. All builders use
``np.random.RandomState`` (stable across hosts/numpy versions) and return
``(x, mask, y)`` with ``mask=None`` for tabular/image.
"""

import json
import os

import numpy as np
import torch

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

_MODEL_CLASS = {
    "tabular": "RoT",
    "text": "RoTText",
    "image": "RoTImage",
}


def _rng(seed):
    return np.random.RandomState(seed)


# --- dataset builders (imported by the mint script and the tests) ---


def data_plot_tabular():
    rng = _rng(0)
    x = rng.rand(40, 4).astype(np.float32)
    y = ((x[:, 0] + x[:, 1]) > 1.0).astype(np.int64)
    return x, None, y


def data_plot_tabular_multi():
    rng = _rng(5)
    x = rng.randn(48, 4).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64) + (x[:, 1] > 0).astype(np.int64)
    return x, None, y


def data_persist(modality):
    from ruleofthumb.text import lengths_to_mask

    rng = _rng(0)
    y = (rng.rand(32) > 0.5).astype(np.int64)
    if modality == "tabular":
        return (rng.rand(32, 4).astype(np.float32), None, y)
    if modality == "text":
        mask = lengths_to_mask(np.array([6, 4] * 16), 6).numpy()
        return (rng.rand(32, 6, 4).astype(np.float32), mask, y)
    return (rng.rand(32, 3, 5, 5).astype(np.float32), None, y)


def data_calib_binary():
    rng = _rng(0)
    return rng.randn(256, 6).astype(np.float32), None, None  # y built by caller from the mechanism


def data_calib_multi():
    rng = _rng(1)
    return rng.randn(256, 6).astype(np.float32), None, None


#: Multiclass mechanism matrix for ``calib_multi`` (single-sourced: the mint
#: script and ``test_calibrated`` both build labels from this).
CALIB_MULTI_W = np.array(
    [
        [2.0, 0.3, -0.2, 0.1, 0.0, 0.1],
        [0.2, 1.8, 0.3, -0.1, 0.1, 0.0],
        [-0.1, 0.2, 1.6, 0.2, -0.1, 0.0],
    ],
    dtype=np.float32,
)


def data_faith_deletion():
    rng = _rng(2)
    x = rng.rand(128, 4).astype(np.float32)
    y = ((x[:, 0] + x[:, 1]) > 1.0).astype(np.int64)
    return x, None, y


def data_faith_pointing():
    rng = _rng(3)
    n = 8
    x = rng.randn(n, 1, 16, 16).astype(np.float32) * 0.2
    y = np.array([i % 2 for i in range(n)], dtype=np.int64)
    x[y == 1, 0, 4:8, 4:8] += 3.0
    return x, None, y


def data_faith_noise():
    rng = _rng(4)
    x = rng.randn(64, 6).astype(np.float32)
    y = ((2 * x[:, 0] - 1.5 * x[:, 1]) > 0).astype(np.int64)
    return x, None, y


def data_ring():
    rng = _rng(0)
    x = rng.uniform(-2, 2, size=(500, 2)).astype(np.float32)
    y = ((x**2).sum(1) < 2.0).astype(np.int64)
    return x, None, y


# --- loader ---


def load_mock(name, device="cpu"):
    """Rebuild a fitted explainer from committed mock weights.

    Reads ``tests/fixtures/<name>.pt`` (raw state dict) plus
    ``<name>_spec.json`` (model class, ``n_classes``, ``sample_shape``,
    ``nonlinear``), constructs the model, loads the weights and wraps it
    in an :class:`Explainer`. Forward passes through loaded weights are
    stable to ulp level across hosts; only the long SGD fit that produced
    them is host-sensitive, and that runs once at mint time.
    """
    from ruleofthumb import Explainer
    from ruleofthumb.core import RoT
    from ruleofthumb.image import RoTImage
    from ruleofthumb.text import RoTText

    with open(os.path.join(FIXTURES, f"{name}_spec.json")) as f:
        spec = json.load(f)
    cls = {"tabular": RoT, "text": RoTText, "image": RoTImage}[spec["model"]]
    model = cls(
        spec["n_classes"],
        tuple(spec["sample_shape"]),
        device=device,
        **({"nonlinear": spec["nonlinear"]} if spec.get("nonlinear") else {}),
    )
    model.load_state_dict(torch.load(os.path.join(FIXTURES, f"{name}.pt"), map_location="cpu", weights_only=True))
    model.eval()
    # mins/maxs are plain attributes (not in the state dict); the mint
    # records them so loaded mocks match fitted models exactly.
    model.mins = torch.as_tensor(spec["mins"])
    model.maxs = torch.as_tensor(spec["maxs"])
    return Explainer(model, spec["model"])


def mock_names():
    """All fixture stems (spec files pin the rebuild parameters)."""
    return sorted(
        f[: -len("_spec.json")]
        for f in os.listdir(FIXTURES)
        if f.endswith("_spec.json")
    )
