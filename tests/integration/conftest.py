"""Shared fixtures for the integration-test tier.

Every fixture loads committed artifacts from ``tests/integration/artifacts/``
(regenerate with ``tests/integration/generate_artifacts.py``). Nothing is
trained or downloaded at test time except HuggingFace/torchvision weights,
which resolve pinned revisions (SST-2 ``SST2_REVISION``, ModernBERT
``DEFAULT_TEXT_REVISION``, MobileNet ``IMAGENET1K_V1``) from the local cache
when present.

Exact reproduction everywhere: the session-autouse ``_deterministic`` fixture
seeds all RNGs and single-threads torch, and every explainer fit runs on
``TEST_DEVICE`` (``"cpu"`` unless ``ROT_TEST_DEVICE`` is set), so macOS/MPS
and Linux/CPU runs compute the same numbers. GPU coverage lives in
``test_device_parity.py`` (opt-in via ``ROT_TEST_DEVICE=mps|cuda``).
"""

import os
import platform
import random

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("sklearn")
import joblib
from _pet_features import compute_pet_features
from cnn import TinyCNN

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

SST2_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SST2_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"

#: Device every integration fit runs on; override locally with
#: ``ROT_TEST_DEVICE=mps`` (or ``=cuda``) — see ``test_device_parity.py``.
TEST_DEVICE = os.environ.get("ROT_TEST_DEVICE", "cpu")


@pytest.fixture(scope="session", autouse=True)
def _deterministic():
    """Seed every RNG and single-thread torch for exact cross-machine repro."""
    seed = int(os.environ.get("ROT_TEST_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    yield


def _require(name):
    path = os.path.join(ARTIFACTS, name)
    if not os.path.exists(path):
        raise RuntimeError(f"missing artifact {path}; regenerate with tests/integration/generate_artifacts.py")
    return path


@pytest.fixture(scope="session")
def tabular_binary():
    """Breast-cancer features + LogisticRegression black box."""
    data = np.load(_require("tabular_binary.npz"))
    lr = joblib.load(_require("breast_cancer_lr.joblib"))["logistic_regression"]
    return {"x": data["x"], "y": data["y"], "coef": lr.coef_[0]}


@pytest.fixture(scope="session")
def tabular_multiclass():
    """Flattened digit images + RandomForest black box (10 classes)."""
    data = np.load(_require("digits_tabular.npz"))
    rf = joblib.load(_require("digits_rf.joblib"))["random_forest"]
    return {"x": data["x"], "y": data["y"], "forest": rf}


def _pet_reference_file():
    """Platform-specific regression anchor filename.

    The 300-epoch pets fit amplifies last-ulp BLAS differences across CPU
    architectures, so each architecture gets its own committed anchor (minted
    with ``mint_pet_reference.py``; x86_64 via the ``mint-pet-reference``
    workflow). ``ROT_PET_REF_ARCH`` overrides detection (testing only).
    """
    machine = os.environ.get("ROT_PET_REF_ARCH", platform.machine()).lower()
    if machine in ("arm64", "aarch64"):
        arch = "arm64"
    elif machine in ("x86_64", "amd64"):
        arch = "x86_64"
    else:
        raise RuntimeError(
            f"no pet reference anchor for architecture {machine!r}; "
            "mint one with tests/integration/mint_pet_reference.py"
        )
    return f"pet_reference_explanations.{arch}.npz"


def _tabular_model_case(stem):
    """Committed dataset + GBM/SVC/MLP black boxes and their predictions."""
    data = np.load(_require(f"{stem}.npz"))
    bundle = joblib.load(_require(f"{stem}_models.joblib"))
    case = {"x": data["x"], "feature_names": bundle["feature_names"]}
    for model_name in ("gbm", "svc", "mlp"):
        case[model_name] = bundle["models"][model_name]
        case[f"y_{model_name}"] = data[f"y_{model_name}"]
    return case


@pytest.fixture(scope="session")
def compas():
    """ProPublica COMPAS two-year recidivism (binary) + GBM/SVC/MLP black boxes."""
    return _tabular_model_case("compas")


@pytest.fixture(scope="session")
def wine():
    """sklearn wine dataset (3 classes) + GBM/SVC/MLP black boxes."""
    return _tabular_model_case("wine")


@pytest.fixture(scope="session")
def pets():
    """Fixed cat/dog JPEGs + recorded GPT-4o-mini labels + reference heatmaps.

    Only raw inputs, labels and the reference *explanations* are committed;
    MobileNet feature maps are never stored and are recomputed afresh in
    :func:`pet_features`. The reference file is platform-specific
    (:func:`_pet_reference_file`); same-architecture runs reproduce it
    bit-for-bit.
    """
    labels = pd.read_csv(_require("pets_labels.csv"))
    try:
        reference = np.load(_require(_pet_reference_file()))["heatmaps"]
    except RuntimeError as e:
        raise RuntimeError(
            f"{e} (mint it with tests/integration/mint_pet_reference.py, "
            "or the mint-pet-reference workflow for other architectures)"
        ) from e
    return {"labels": labels, "reference": reference, "images_dir": os.path.join(ARTIFACTS, "pet_images")}


@pytest.fixture(scope="session")
def pet_features(pets):
    """Live MobileNetV3-Small feature maps ``(N, 576, 7, 7)`` for the pet set."""
    pytest.importorskip("torchvision")

    labels = pets["labels"]
    features = compute_pet_features(os.path.join(ARTIFACTS, "pet_images"), labels["filename"])
    y_gpt = labels["gpt_label"].eq("dog").to_numpy().astype(np.int64)
    ground_truth = labels["ground_truth"].eq("dog").to_numpy().astype(np.int64)
    return {"features": features, "y_gpt": y_gpt, "ground_truth": ground_truth}


@pytest.fixture(scope="session")
def digit_features_multiclass():
    """Live MobileNetV3-Small feature maps ``(500, 576, 7, 7)`` for the digit set.

    The same 500 8x8 digits as ``image_multiclass`` are upscaled to RGB and
    passed through the pretrained backbone live on every run (weights from
    the local cache; nothing intermediate is committed). Labels are the
    committed TinyCNN 10-class predictions, so the RoT surrogate is still
    the only model fitted at test time. ``N=500`` exceeds the 49 spatial
    positions per map, keeping the shared-weight fit overdetermined.
    """
    pytest.importorskip("torchvision")
    from PIL import Image
    from torchvision import models as tv_models

    data = np.load(_require("digits_image.npz"))
    x = data["x_multi"].astype(np.float32)
    y = data["y_multi"].astype(np.int64)
    weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    backbone = tv_models.mobilenet_v3_small(weights=weights).eval()
    transform = weights.transforms()
    batch = torch.stack(
        [
            transform(Image.fromarray((x[i, 0] * 255).clip(0, 255).astype(np.uint8), mode="L").convert("RGB"))
            for i in range(len(x))
        ]
    )
    with torch.no_grad():
        features = backbone.features(batch).numpy().astype(np.float32)
    return {"features": features, "y": y}


@pytest.fixture(scope="session")
def image_multiclass():
    """Digit images ``(N, 1|3, 8, 8)`` + TinyCNN black boxes (2 and 10 classes).

    Includes a coordinate-augmented 10-class view: intensity plus
    intensity-gated row/col grids, so pooling retains mass and center-of-mass.
    """
    data = np.load(_require("digits_image.npz"))
    state = torch.load(_require("cnn_multiclass.pt"), map_location="cpu", weights_only=True)
    cnn_multi = TinyCNN(n_classes=10)
    cnn_multi.load_state_dict(state)
    cnn_multi.eval()
    state = torch.load(_require("cnn_binary.pt"), map_location="cpu", weights_only=True)
    cnn_binary = TinyCNN(n_classes=2)
    cnn_binary.load_state_dict(state)
    cnn_binary.eval()
    state = torch.load(_require("cnn_multiclass_coords.pt"), map_location="cpu", weights_only=True)
    cnn_coords = TinyCNN(n_classes=10, in_channels=3)
    cnn_coords.load_state_dict(state)
    cnn_coords.eval()
    x = data["x_multi"]
    with torch.no_grad():
        consistent_multi = bool((cnn_multi(torch.from_numpy(x)).argmax(1).numpy() == data["y_multi"]).all())
        consistent_binary = bool((cnn_binary(torch.from_numpy(x)).argmax(1).numpy() == data["y_binary"]).all())
        consistent_coords = bool(
            (cnn_coords(torch.from_numpy(data["x_coords"])).argmax(1).numpy() == data["y_coords"]).all()
        )
    return {
        "x": x,
        "y": data["y_multi"],
        "y_binary": data["y_binary"],
        "x_coords": data["x_coords"],
        "y_coords": data["y_coords"],
        "cnn_multi": cnn_multi,
        "cnn_binary": cnn_binary,
        "consistent_multi": consistent_multi,
        "consistent_binary": consistent_binary,
        "consistent_coords": consistent_coords,
    }


@pytest.fixture(scope="session")
def text_sst2():
    """Fixed film reviews encoded with the cached SST-2 transformer."""
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(SST2_NAME, revision=SST2_REVISION)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(SST2_NAME, revision=SST2_REVISION)
    model.eval()

    with open(_require("reviews.txt")) as f:
        texts = [line.strip() for line in f if line.strip()]
    encoded = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=48)
    attention_mask = encoded["attention_mask"].to(torch.bool)
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)
    logits = out.logits
    probabilities = torch.softmax(logits, dim=-1)
    embeddings = out.hidden_states[-1].numpy().astype(np.float32)
    return {
        "tokenizer": tokenizer,
        "encoded": encoded,
        "texts": texts,
        "embeddings": embeddings,
        "attention_mask": attention_mask,
        "y": logits.argmax(-1).numpy().astype(np.int64),
        "confidence": probabilities.max(-1).values,
        "n_classes": logits.shape[-1],
        "model_name": SST2_NAME,
    }
