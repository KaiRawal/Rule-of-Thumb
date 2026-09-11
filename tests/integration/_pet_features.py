"""Single source of truth for pet MobileNet feature maps.

Both the ``pet_features`` fixture (``conftest.py``) and
``mint_pet_reference.py`` compute through :func:`compute_pet_features`, so a
minted reference anchor is bit-identical to what the test computes on the
same machine and architecture.
"""

import os

import numpy as np
import torch


def compute_pet_features(images_dir, filenames):
    """MobileNetV3-Small ``IMAGENET1K_V1`` feature maps for the pet JPEGs.

    Pinned weights, CPU, ``no_grad`` — bit-identical on a given CPU
    architecture for fixed library versions (see ``requirements.txt``).
    Returns ``(N, 576, 7, 7)`` float32 numpy.
    """
    from PIL import Image
    from torchvision import models as tv_models

    weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    backbone = tv_models.mobilenet_v3_small(weights=weights).eval()
    transform = weights.transforms()
    batch = torch.stack(
        [transform(Image.open(os.path.join(images_dir, name)).convert("RGB")) for name in filenames]
    )
    with torch.no_grad():
        return backbone.features(batch).numpy().astype(np.float32)
