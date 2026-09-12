"""External-cache helpers for the human-annotation integration tests.

The HateXPlain / MIT1003 cache lives in ``tests/integration/external_cache/``
(gitignored; populate with ``tests/integration/fetch_external.py`` — CI runs
it as a setup step). Tests skip cleanly when the cache is absent; they never
download or train black boxes at test time.
"""

import json
import os

import numpy as np
import pytest
import torch

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_cache")


def require_cache(name):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        pytest.skip(f"external cache file {name} absent; run tests/integration/fetch_external.py")
    return path


def load_hx():
    import joblib

    with open(require_cache("hx_full.json")) as f:
        full = json.load(f)
    box = joblib.load(require_cache("hx_box.joblib"))
    ybb = {s: np.load(require_cache(f"hx_ybb_{s}.npy")) for s in ("train", "val", "test")}
    return {"full": full, "vec": box["vec"], "bb": box["bb"], "ybb": ybb}


def rebuild_text_explainer(artifacts, pt_name="hx_rot_text.pt", spec_name="hx_rot_spec.json"):
    from ruleofthumb.explain import Explainer
    from ruleofthumb.text import RoTText

    with open(os.path.join(artifacts, spec_name)) as f:
        spec = json.load(f)
    model = RoTText(spec["n_classes"], tuple(spec["sample_shape"]), device="cpu")
    model.load_state_dict(torch.load(os.path.join(artifacts, pt_name), map_location="cpu", weights_only=True))
    model.eval()
    return Explainer(model, "text"), spec


def rebuild_image_explainer(artifacts, stem):
    from ruleofthumb.explain import Explainer
    from ruleofthumb.image import RoTImage

    with open(os.path.join(artifacts, f"{stem}_spec.json")) as f:
        spec = json.load(f)
    model = RoTImage(spec["n_classes"], tuple(spec["sample_shape"]), device="cpu")
    model.load_state_dict(torch.load(os.path.join(artifacts, f"{stem}.pt"), map_location="cpu", weights_only=True))
    model.eval()
    return Explainer(model, "image"), spec
