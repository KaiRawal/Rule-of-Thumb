"""Integration tests: the legacy GPT cat-vs-dog experiment, miniaturized.

Mirrors ``ExplanationExampleRemote/run.py``: MobileNetV3-Small feature maps of
cat/dog JPEGs are explained through a binary :func:`ruleofthumb.fit_image`
surrogate fitted on **GPT-4o-mini labels** (the black box being explained is
the vision-language model's behaviour, not the ground truth).

Feature maps are recomputed live on every run from the committed raw JPEGs;
nothing intermediate is cached. The committed reference explanations act as
regression anchors only.
"""

import numpy as np
from _helpers import rot_accuracy

from ruleofthumb import fit_image

SEED = 0


def _fit_pets(features, y_gpt):
    return fit_image(y_gpt, features, epochs=300, batch_size=5000, learning_rate=0.05, seed=SEED)


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
    """Freshly computed saliency must reproduce the committed reference."""
    features, y_gpt = pet_features["features"], pet_features["y_gpt"]
    heatmaps = _fit_pets(features, y_gpt).get_explanation(features)
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
