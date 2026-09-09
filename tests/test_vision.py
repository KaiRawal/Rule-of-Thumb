"""Unit tests for the image backbone path (Bug 12).

All backbone tests use a tiny stub conv module — no weight downloads, fully
offline and deterministic. The live default backbone is exercised in
integration only.
"""

import numpy as np
import pytest
import torch


class _StubBackbone(torch.nn.Module):
    """Stride-8 single conv: 16x16 RGB -> 4-channel 2x2 maps, constant weights."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=8, padding=1)
        torch.nn.init.constant_(self.conv.weight, 0.1)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


def _write_png(path, height, width, colour=(120, 60, 200)):
    from PIL import Image

    Image.new("RGB", (width, height), colour).save(path)


@pytest.fixture
def map_png_paths(tmp_path):
    small = tmp_path / "small.png"
    big = tmp_path / "big.png"
    _write_png(small, 16, 16)
    _write_png(big, 12, 20)
    return [str(small), str(big)]


def test_embed_images_stub_shapes_and_mask(map_png_paths):
    from ruleofthumb.vision import embed_images

    out = embed_images(map_png_paths, backbone=_StubBackbone())
    assert out.maps.shape == (2, 4, 2, 3)
    assert out.maps.dtype == np.float32
    assert out.mask.shape == (2, 2, 3)
    assert out.mask.dtype == bool
    assert int(out.mask[0].sum()) == 2 * 2  # 16x16 image -> 2x2 real cells
    assert int(out.mask[1].sum()) == 2 * 3  # 12x20 image -> 2x3 real cells


def test_fit_image_stub_backbone_end_to_end(map_png_paths):
    from ruleofthumb import fit_image

    y = np.array([0, 1])
    exp = fit_image(y, map_png_paths, backbone=_StubBackbone(), epochs=2, batch_size=2, learning_rate=0.05)
    assert exp.backbone is None  # custom modules record no id
    imp = exp.get_explanation(map_png_paths)
    assert imp.shape == (2, 2, 3)


def test_fit_image_backbone_none_keeps_pixels(map_png_paths):
    from ruleofthumb import fit_image

    y = np.array([0, 1])
    exp = fit_image(y, map_png_paths, backbone=None, epochs=2, batch_size=2, learning_rate=0.05)
    assert exp.backbone is None
    assert exp.get_explanation(map_png_paths).shape == (2, 16, 20)


def test_backbone_transform_conflict_rejected(map_png_paths):
    from ruleofthumb import fit_image

    with pytest.raises(ValueError, match="transform"):
        fit_image(
            np.array([0, 1]), map_png_paths, backbone=_StubBackbone(), transform=lambda pil: pil, epochs=1
        )


def test_save_load_preserves_backbone_id(map_png_paths, tmp_path):
    from ruleofthumb import fit_image, load_explainer
    from ruleofthumb.vision import embed_images

    y = np.array([0, 1])
    exp = fit_image(y, map_png_paths, backbone=_StubBackbone(), epochs=2, batch_size=2, learning_rate=0.05)
    path = tmp_path / "maps.rotx"
    exp.save(str(path))
    loaded = load_explainer(str(path))
    assert loaded.backbone is None
    # reloaded explainers consume arrays, like strings today
    maps = embed_images(map_png_paths, backbone=_StubBackbone()).maps
    assert np.allclose(exp.get_explanation(maps), loaded.get_explanation(maps))
