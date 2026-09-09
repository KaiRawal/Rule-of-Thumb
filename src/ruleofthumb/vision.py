"""Image backbone embeddings (Bug 12).

Text has a bundled default embedding; images get the same treatment here:
:func:`embed_images` resolves file paths through a frozen torchvision
backbone into ``(N, C, h, w)`` feature maps with a validity mask pooled to
the map grid, mirroring :func:`ruleofthumb.embed_texts`. Raw-pixel fitting
stays available via ``backbone=None`` / ``backbone=False`` in
:func:`ruleofthumb.fit_image`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

from ruleofthumb.image import pad_images

#: Default backbone used for image file paths (frozen, eval mode).
DEFAULT_IMAGE_MODEL = "mobilenet_v3_small"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclasses.dataclass(frozen=True)
class ImageEmbeddings:
    """Feature maps decoded from files, ready for :func:`ruleofthumb.fit_image`.

    Attributes:
        maps: ``(N, channels, height, width)`` float32 maps, zero-padded
            beyond each sample's real region.
        mask: ``(N, height, width)`` boolean validity mask pooled to the map
            grid (``True`` marks real cells); pass as ``mask=`` to
            array-based entry points.
    """

    maps: np.ndarray
    mask: np.ndarray


def _mobilenet_v3_small_features():
    """Frozen MobileNetV3-Small feature trunk (downloads weights once, torchvision cache)."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    features = model.features
    features.eval()
    return features


_REGISTERED = {"mobilenet_v3_small": _mobilenet_v3_small_features}


def _resolve_backbone(backbone):
    """Reduce a ``backbone=`` spec to a feature trunk module."""
    if isinstance(backbone, torch.nn.Module):
        backbone.eval()
        return backbone
    if isinstance(backbone, str) and backbone in _REGISTERED:
        return _REGISTERED[backbone]()
    raise ValueError(f"unknown backbone {backbone!r}; expected one of {sorted(_REGISTERED)} or a torch module")


def _preprocess(pil, size):
    """Decode to ImageNet-normalised ``(C, H, W)`` tensor, honouring ``size=``."""
    from torchvision.transforms import functional as tf

    if size is not None:
        height, width = int(size[0]), int(size[1])
        pil = tf.center_crop(tf.resize(pil, min(height, width)), [height, width])
    tensor = tf.to_tensor(pil)
    return tf.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)


def _pool_mask(pixel_mask, height, width):
    """Max-pool a ``(H, W)`` validity mask to the map grid; a cell is real if any pixel is."""
    pooled = torch.nn.functional.adaptive_max_pool2d(
        torch.as_tensor(pixel_mask).to(torch.float32)[None, None], (height, width)
    )
    return (pooled[0, 0] > 0.5).numpy()


def embed_images(
    paths: Sequence[str], *, backbone: str | torch.nn.Module = DEFAULT_IMAGE_MODEL, size: tuple[int, int] | None = None
) -> ImageEmbeddings:
    """Embed image files into backbone feature maps with a validity mask.

    Args:
        paths: list of image file paths (PNG / JPEG / ...).
        backbone: registered name (``"mobilenet_v3_small"``) or a torch
            module mapping ``(N, 3, H, W)`` ImageNet-normalised RGB to
            ``(N, C, h, w)`` maps. Custom modules receive normalised
            tensors and run frozen in eval mode under ``no_grad``.
        size: optional ``(height, width)``; when given, every image is
            resized (shorter edge) and centre-cropped before embedding.
            Without it, native sizes are kept and maps are zero-padded.

    Returns:
        :class:`ImageEmbeddings` with float32 maps and a boolean
        ``(N, h, w)`` validity mask pooled from the pixel grid.
    """
    from PIL import Image

    if len(paths) == 0:
        raise ValueError("paths must be non-empty")
    trunk = _resolve_backbone(backbone)

    maps, masks = [], []
    with torch.no_grad():
        for path in paths:
            with Image.open(path) as image:
                pil = image.convert("RGB")
            pixels = _preprocess(pil, size)
            pixel_mask = torch.ones(pixels.shape[1:])
            features = trunk(pixels[None].cpu())
            _, _, height, width = features.shape
            maps.append(features[0].cpu())
            masks.append(torch.from_numpy(_pool_mask(pixel_mask.numpy(), height, width)))
    channels = int(maps[0].shape[0])
    if any(m.shape[0] != channels for m in maps):
        raise ValueError("backbone produced inconsistent channel counts")
    padded, _ = pad_images(maps)
    mask_batch = torch.zeros(padded.shape[:1] + padded.shape[2:], dtype=torch.bool)
    for i, m in enumerate(masks):
        mask_batch[i, : m.shape[0], : m.shape[1]] = m
    return ImageEmbeddings(maps=padded.numpy().astype(np.float32), mask=mask_batch.numpy())
