"""Image RoT variant.

Port of ``RoT_image`` from the original experiment code
(``ExplanationExampleRemote/rot_class.py`` and
``AdversarialAttack/rot_class.py``).

Since v0.2 batches of mixed-size images are supported by padding: pass a
boolean validity ``mask`` of shape ``(N, H, W)`` alongside the padded batch
(see :func:`pad_images`). Because the importance weights are spatially shared,
scoring individual unpadded samples one at a time remains fully supported and
needs no mask.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from ruleofthumb.core import RoT


@dataclasses.dataclass(frozen=True)
class ImageBatch:
    """Images decoded from files, ready for :func:`ruleofthumb.fit_image`.

    Attributes:
        images: ``(N, channels, height, width)`` float32 array, zero-padded
            beyond each sample's real region.
        mask: ``(N, height, width)`` boolean validity mask (``True`` marks
            real pixels); pass as ``mask=`` to array-based entry points.
    """

    images: np.ndarray
    mask: np.ndarray


def load_images(
    paths: Sequence[str], *, size: tuple[int, int] | None = None, transform: Callable[..., Any] | None = None
) -> ImageBatch:
    """Decode image files into a rectangular batch with a validity mask.

    Args:
        paths: list of image file paths (PNG / JPEG / ...).
        size: optional ``(height, width)``; when given, every image is
            resized (shorter edge) and centre-cropped to that common size and
            the mask is all-true. When ``None`` (default), native sizes are
            kept and smaller images are zero-padded with real-region masks.
        transform: optional callable overriding the default pipeline entirely
            (PIL Image -> ``(C, H, W)`` tensor), e.g. a torchvision weights
            transform including normalisation; output must be uniform so the
            mask is all-true.

    Returns:
        :class:`ImageBatch` with float32 RGB images scaled to ``[0, 1]``
        (unless ``transform`` replaces the scaling).
    """
    from PIL import Image

    if len(paths) == 0:
        raise ValueError("paths must be non-empty")

    tensors = []
    for path in paths:
        with Image.open(path) as image:
            pil = image.convert("RGB")
        if transform is not None:
            tensors.append(torch.as_tensor(transform(pil)).to(torch.float32))
        elif size is not None:
            tensors.append(_resize_and_crop(pil, size))
        else:
            tensors.append(torch.from_numpy(np.array(pil)).permute(2, 0, 1).to(torch.float32) / 255.0)

    channels = int(tensors[0].shape[0])
    if any(t.shape[0] != channels for t in tensors):
        raise ValueError("all images must have the same number of channels")
    if any(t.shape[1:] != tensors[0].shape[1:] for t in tensors):
        padded, mask = pad_images(tensors)
        return ImageBatch(images=padded.numpy().astype(np.float32), mask=mask.numpy())
    batch = torch.stack(tensors)
    return ImageBatch(images=batch.numpy().astype(np.float32), mask=np.ones(batch.shape[:1] + batch.shape[2:], dtype=bool))


def _resize_and_crop(pil, size):
    """Resize the shorter edge to ``size`` and centre-crop to ``(h, w)``."""
    from torchvision.transforms import functional as tf

    height, width = int(size[0]), int(size[1])
    resized = tf.resize(pil, min(height, width))
    return tf.center_crop(tf.to_tensor(resized), [height, width])


def pad_images(images, pad_value=0.0):
    """Pad a list of ``(C, H_i, W_i)`` images into one rectangular batch.

    Returns ``(padded, mask)`` where ``padded`` has shape
    ``(N, C, max(H_i), max(W_i))`` padded with ``pad_value``, and ``mask`` is
    a boolean ``(N, max(H_i), max(W_i))`` validity mask to pass on to
    :meth:`RoTImage.fit` / :meth:`~ruleofthumb.core.RoT.score` /
    ``importance``.
    """
    imgs = [torch.as_tensor(x) for x in images]
    n = len(imgs)
    channels = int(imgs[0].shape[0])
    height = int(max(x.shape[1] for x in imgs))
    width = int(max(x.shape[2] for x in imgs))
    padded = torch.full((n, channels, height, width), float(pad_value), dtype=imgs[0].dtype)
    mask = torch.zeros((n, height, width), dtype=torch.bool)
    for i, x in enumerate(imgs):
        h, w = int(x.shape[1]), int(x.shape[2])
        padded[i, :, :h, :w] = x
        mask[i, :h, :w] = True
    return padded, mask


class RoTImage(RoT):
    """Share importance between spatial locations.

    Definitions: `C` is the channel count per pixel (1 for grey, 576 for
    MobileNetV3-Small maps); `H`/`W` are height/width in pixels. The learned
    slopes `a` and offsets `b` have one row per channel and that row is
    reused at every pixel, so the score is linear on per-channel spatial
    sums. Raw `C=1` inputs pool to total ink mass (10-class digits cap near
    the majority baseline); rich pretrained feature maps restore strong
    multiclass fidelity (see README Limitations and the rich-backbone
    integration test). Assumes datapoints are in the form channel x width x
    height. An optional boolean ``mask`` of shape ``(N, H, W)`` marks real
    (non-padding) pixels; masked-out pixels receive exactly zero importance
    and are excluded from fit bounds and scores.
    """

    def __init__(self, classes, sample_shape, dropout_rate=0.5, use_BCE_loss=False, device=None, nonlinear=None):
        super().__init__(classes, sample_shape, dropout_rate, use_BCE_loss, no_a_b=True, device=device, nonlinear=nonlinear)
        self.a = nn.Parameter(torch.zeros((classes, sample_shape[0]), requires_grad=True, device=self.device))
        self.b = nn.Parameter(torch.zeros((classes, sample_shape[0]), requires_grad=True, device=self.device))
        self.weights = (self.a, self.b, self.g)

    def importance(self, points, mask=None):
        # Convolutional form.
        # Treat all spatial locations given by last two axis the same
        points = torch.as_tensor(points, device=self.device)
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
        imp = self.a[None, :, :, None, None] * (self._respond(points)[:, None] + self.b[None, :, :, None, None])
        if mask is None:
            return imp
        return imp * mask.unsqueeze(1).unsqueeze(1).to(imp.dtype)

    def stochastic_importance(self, points, mask=None):
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
        imp = self.importance(points, mask=mask)
        keep = (
            torch.rand(points.shape[0], *points.shape[2:], device=self.device) > self.dropout_rate
        ).float()
        if mask is not None:
            keep = keep * mask.to(keep.dtype)
        return keep.unsqueeze(1).unsqueeze(1) * imp

    def _reduce_to_units(self, imp):
        """Reveal units are pixels: aggregate signed importance over channels."""
        return imp.sum(dim=2)

    def fit(
        self,
        points,
        classifier_response,
        epochs,
        batch_size,
        lr=1e-4,
        mask=None,
        pretrain_epochs=5,
        weight_decay=0.01,
        seed=None,
    ):
        assert points.shape[0] == classifier_response.shape[0]
        assert points.shape[1] == self.a.shape[1]
        points = torch.as_tensor(points, device=self.device)
        classifier_response = torch.as_tensor(classifier_response, device=self.device)
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
        if seed is not None:
            torch.manual_seed(seed)
        if mask is None:
            upper = points.amax(dim=(0, 2, 3))
            lower = points.amin(dim=(0, 2, 3))
            mean = points.mean(dim=(0, 2, 3))
        else:
            m = mask[:, None].to(points.dtype)
            upper = points.masked_fill(m == 0, float("-inf")).amax(dim=(0, 2, 3))
            lower = points.masked_fill(m == 0, float("+inf")).amin(dim=(0, 2, 3))
            mean = (points * m).sum(dim=(0, 2, 3)) / m.sum(dim=(0, 2, 3)).clamp(min=1)
        self.mins = -upper
        self.maxs = -lower
        with torch.no_grad():
            self.b.copy_(-mean.view(1, -1).expand_as(self.b))

        optimiser = torch.optim.AdamW(self.parameters(), lr=lr)
        drop_out = self.dropout_rate
        self.dropout_rate = 0
        self.training_loop(self.loss, points, classifier_response, optimiser, pretrain_epochs, batch_size, mask=mask)
        self.dropout_rate = drop_out
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.training_loop(self.loss, points, classifier_response, optimiser, epochs, batch_size, mask=mask)
