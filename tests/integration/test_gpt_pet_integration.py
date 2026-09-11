"""Integration tests: the legacy GPT cat-vs-dog experiment, miniaturized.

Mirrors ``ExplanationExampleRemote/run.py``: MobileNetV3-Small feature maps of
cat/dog JPEGs are explained through a binary :func:`rot.fit_image`
surrogate fitted on **GPT-4o-mini labels** (the black box being explained is
the vision-language model's behaviour, not the ground truth).

Feature maps are recomputed live on every run from the committed raw JPEGs.
The fitted surrogate weights and their reference explanations are committed
(`pet_rot_state.pt`, `pet_reference_explanations.npz` — minted with
`mint_pet_weights.py`); the anchor test loads them without training, while
accuracy/direction tests fit live to guard the training path.
"""

import os

import numpy as np
import torch
from _helpers import rot_accuracy

from ruleofthumb import Explainer, fit_image
from ruleofthumb.image import RoTImage

SEED = 0

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _fit_pets(features, y_gpt):
    # Live fits guard the training path (accuracy/direction tests below).
    # The anchor comparison loads committed weights instead: the 300-epoch
    # fit amplifies last-ulp BLAS differences across hosts, so fresh fits
    # cannot reproduce a committed anchor exactly on every machine.
    return fit_image(y_gpt, features, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED, device="cpu")


def _load_pets_explainer():
    """Explainer rebuilt from the committed fitted weights (no training).

    Raw ``state_dict``, deliberately not ``.rotx``: ``load_explainer``
    enforces an exact package-version match, which would break this fixture
    on every release (the ``.rotx`` path is covered by
    ``test_image_round_trip``). Heatmaps are a pure forward pass of these
    weights, hence stable to ulp level on every host.
    """
    model = RoTImage(2, (576,), device="cpu")
    model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "pet_rot_state.pt"), map_location="cpu", weights_only=True))
    return Explainer(model, "image")


def test_feature_maps_shape(pet_features):
    features = pet_features["features"]
    assert features.shape == (20, 576, 7, 7)
    assert np.isfinite(features).all()


def test_gpt_labels_are_accurate_and_balanced(pets, pet_features):
    labels = pets["labels"]
    assert set(labels["gpt_label"]) == {"cat", "dog"}
    gpt_accuracy = float((pet_features["y_gpt"] == pet_features["ground_truth"]).mean())
    assert gpt_accuracy >= 0.8  # GPT-4o-mini scored 100% on this fixed subset


def test_rot_surrogate_accuracy_against_gpt_labels(pet_features):
    features, y_gpt = pet_features["features"], pet_features["y_gpt"]
    exp = _fit_pets(features, y_gpt)
    assert rot_accuracy(exp, features, y_gpt) >= 0.85


def test_heatmaps_match_reference_explanations(pets, pet_features):
    """Committed weights reproduce the committed reference (no training).

    Heatmaps are a pure forward pass of the loaded weights, so this is
    stable to ulp level on every host — unlike refitting, which diverges
    across hosts. Training-path coverage stays with the live fits below.
    """
    features, y_gpt = pet_features["features"], pet_features["y_gpt"]
    exp = _load_pets_explainer()
    assert float((exp.predict(torch.from_numpy(features)).cpu().numpy() == y_gpt).mean()) == 1.0
    heatmaps = exp.get_explanation(features)
    reference = pets["reference"]
    assert heatmaps.shape == reference.shape == (20, 7, 7)

    correlations = [np.corrcoef(h.ravel(), r.ravel())[0, 1] for h, r in zip(heatmaps, reference)]
    assert np.nanmin(correlations) >= 0.95
    assert float(np.mean(correlations)) >= 0.99

    # signed class-"dog" importances: both directions occur
    assert (heatmaps > 0).any() and (heatmaps < 0).any()


def test_dog_images_highlight_the_dog_direction(pets, pet_features):
    """GPT-"dog" images carry more positive dog-mass than GPT-"cat" images."""
    features, y_gpt = pet_features["features"], pet_features["y_gpt"]
    heatmaps = _fit_pets(features, y_gpt).get_explanation(features)
    dog_mass = heatmaps.sum(axis=(1, 2))

    mean_dog = float(dog_mass[y_gpt == 1].mean())
    mean_cat = float(dog_mass[y_gpt == 0].mean())
    assert mean_dog > mean_cat
    # near-perfect separation: dog-mass ranks the GPT labels almost alone
    assert np.corrcoef(dog_mass, y_gpt)[0, 1] >= 0.9
