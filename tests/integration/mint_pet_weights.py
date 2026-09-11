"""Mint the committed pets fitted weights + reference heatmaps.

Usage (repo root, venv only)::

    .venv/bin/python tests/integration/mint_pet_weights.py [--check]

Trains the exact ``_fit_pets`` fit from ``test_gpt_pet_integration`` once
(CPU, seeded, single-threaded) on the committed pet JPEGs/labels and writes:

- ``artifacts/pet_rot_state.pt`` — the fitted ``RoTImage`` state dict (raw
  weights, deliberately not ``.rotx``: ``load_explainer`` enforces an exact
  package-version match, which would break this fixture on every release;
  the ``.rotx`` path is covered separately by ``test_image_round_trip``);
- ``artifacts/pet_reference_explanations.npz`` — heatmaps from those weights;
- ``manifest.json`` entries for both.

``--check`` reloads the committed weights, recomputes heatmaps through a
freshly constructed model, and verifies they match both the committed
reference and a fresh live fit on this machine (exits nonzero on mismatch).
"""

import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
STATE_NAME = "pet_rot_state.pt"
REF_NAME = "pet_reference_explanations.npz"


def _seed_everything():
    seed = int(os.environ.get("ROT_TEST_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _live_inputs():
    from PIL import Image
    from torchvision import models as tv_models

    _seed_everything()
    labels = pd.read_csv(os.path.join(ARTIFACTS, "pets_labels.csv"))
    weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    backbone = tv_models.mobilenet_v3_small(weights=weights).eval()
    transform = weights.transforms()
    batch = torch.stack(
        [transform(Image.open(os.path.join(ARTIFACTS, "pet_images", name)).convert("RGB")) for name in labels["filename"]]
    )
    with torch.no_grad():
        features = backbone.features(batch).numpy().astype(np.float32)
    y_gpt = labels["gpt_label"].eq("dog").to_numpy().astype(np.int64)
    return features, y_gpt


def _build_model():
    from ruleofthumb.image import RoTImage

    return RoTImage(2, (576,), device="cpu")


def mint():
    from test_gpt_pet_integration import _fit_pets

    features, y_gpt = _live_inputs()
    exp = _fit_pets(features, y_gpt)
    state_path = os.path.join(ARTIFACTS, STATE_NAME)
    torch.save(exp.model.state_dict(), state_path)
    heatmaps = exp.get_explanation(features).astype(np.float32)
    assert heatmaps.shape == (20, 7, 7)
    ref_path = os.path.join(ARTIFACTS, REF_NAME)
    np.savez_compressed(ref_path, heatmaps=heatmaps)

    from generate_artifacts import record

    manifest_path = os.path.join(ARTIFACTS, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    record(manifest, STATE_NAME)
    record(manifest, REF_NAME, {"heatmaps": heatmaps})
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    for path in (state_path, ref_path):
        print(f"minted {path}")
        print(f"sha256: {hashlib.sha256(Path(path).read_bytes()).hexdigest()}")


def check():
    from ruleofthumb import Explainer

    features, y_gpt = _live_inputs()
    committed = np.load(os.path.join(ARTIFACTS, REF_NAME))["heatmaps"]

    model = _build_model()
    model.load_state_dict(torch.load(os.path.join(ARTIFACTS, STATE_NAME), map_location="cpu", weights_only=True))
    loaded = Explainer(model, "image").get_explanation(features)
    assert np.array_equal(loaded, committed), "committed weights do not reproduce the committed reference here"

    from test_gpt_pet_integration import _fit_pets

    fresh = _fit_pets(features, y_gpt).get_explanation(features)
    assert np.array_equal(fresh, committed), "fresh live fit differs from the committed reference on this machine"

    accuracy = float((Explainer(model, "image").predict(torch.from_numpy(features)).cpu().numpy() == y_gpt).mean())
    print(f"OK: loaded weights reproduce the reference bit-for-bit; loaded-model accuracy {accuracy:.4f}")


def main():
    if "--check" in sys.argv[1:]:
        check()
    else:
        mint()


if __name__ == "__main__":
    main()
