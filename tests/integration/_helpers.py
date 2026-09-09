"""Shared helpers for the integration tier."""

import os

import numpy as np
import torch

#: Device every integration fit runs on: ``"cpu"`` unless ``ROT_TEST_DEVICE``
#: is set (e.g. ``ROT_TEST_DEVICE=mps`` for local GPU smoke coverage).
TEST_DEVICE = os.environ.get("ROT_TEST_DEVICE", "cpu")


def rot_accuracy(explainer, x_numpy, y_numpy):
    """Accuracy of the RoT surrogate's own predicted classes vs black-box labels."""
    predictions = explainer.predict(torch.from_numpy(np.asarray(x_numpy))).cpu().numpy()
    return float((predictions == np.asarray(y_numpy)).mean())
