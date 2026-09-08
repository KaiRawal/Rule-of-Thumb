"""Shared helpers for the integration tier."""

import numpy as np
import torch


def rot_accuracy(explainer, x_numpy, y_numpy):
    """Accuracy of the RoT surrogate's own predicted classes vs black-box labels."""
    predictions = explainer.predict(torch.from_numpy(np.asarray(x_numpy))).cpu().numpy()
    return float((predictions == np.asarray(y_numpy)).mean())
