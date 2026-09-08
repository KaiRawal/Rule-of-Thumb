import numpy as np
import pytest
import torch

from ruleofthumb.image import RoTImage, pad_images


@pytest.fixture
def image_data():
    rng = np.random.RandomState(0)
    # N x channels x width x height
    x = rng.randn(8, 3, 4, 4).astype(np.float32)
    y = torch.randint(0, 2, (8,))
    return x, y


def test_rot_image_importance_shape(image_data):
    x, _ = image_data
    model = RoTImage(2, (x.shape[1],))
    imp = model.importance(torch.from_numpy(x))
    assert tuple(imp.shape) == (8, 2, 3, 4, 4)


def test_rot_image_fit_and_score(image_data):
    x, y = image_data
    model = RoTImage(2, (x.shape[1],))
    model.fit(torch.from_numpy(x), y, epochs=4, batch_size=8, lr=0.05)
    score = model.score(torch.from_numpy(x))
    assert score.shape == (8, 2)


def test_mask_zeroes_padded_pixels():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    mask[:, :4, :5] = True
    model = RoTImage(2, (3,))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)
    imp = model.importance(x, mask=mask)
    assert torch.all(imp[:, :, :, 5:, :] == 0)
    assert torch.all(imp[:, :, 4:, :, :] == 0)
    assert torch.any(imp != 0)


def test_mixed_size_batch_matches_per_sample_loop():
    """Padded+masked scoring equals scoring each unpadded sample individually."""
    torch.manual_seed(0)
    rng = np.random.RandomState(1)
    small = rng.randn(1, 3, 4, 4).astype(np.float32)
    big = rng.randn(1, 3, 6, 7).astype(np.float32)

    padded, mask = pad_images([small[0], big[0]], pad_value=0.0)
    model = RoTImage(2, (3,))
    torch.nn.init.normal_(model.a, std=1.0)
    torch.nn.init.normal_(model.b, std=1.0)

    batch_scores = model.score(padded, mask=mask)
    loop_small = model.score(torch.from_numpy(small))[0]
    loop_big = model.score(torch.from_numpy(big))[0]

    assert torch.allclose(batch_scores[0], loop_small, atol=1e-5)
    assert torch.allclose(batch_scores[1], loop_big, atol=1e-5)


def test_fit_with_padded_mask():
    rng = np.random.RandomState(0)
    imgs = [rng.randn(3, h, w).astype(np.float32) for h, w in [(4, 4), (6, 5), (5, 6)]]
    labels = torch.randint(0, 2, (3,))
    padded, mask = pad_images(imgs)
    model = RoTImage(2, (3,))
    model.fit(padded, labels, epochs=4, batch_size=3, lr=0.05, mask=mask)
    assert len(model.training_loss) == 4

    scores = model.score(padded, mask=mask)
    assert scores.shape == (3, 2)


def test_pad_images_utility():
    rng = np.random.RandomState(0)
    imgs = [rng.randn(3, 4, 4).astype(np.float32), rng.randn(3, 6, 5).astype(np.float32)]
    padded, mask = pad_images(imgs, pad_value=9.0)
    assert tuple(padded.shape) == (2, 3, 6, 5)
    assert tuple(mask.shape) == (2, 6, 5)
    assert torch.all(mask[0, :4, :4])
    assert not mask[0, 4:, :].any()
    assert torch.all(padded[0, :, 4:, :] == 9.0)
    assert torch.equal(padded[1, :, :6, :5], torch.from_numpy(imgs[1]))


def test_facade_binary_pipeline_with_untrained_conv_black_box():
    """Binary image explanations against a real (untrained) conv black box."""
    from ruleofthumb import fit_image

    torch.manual_seed(0)
    black_box = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 8 * 8, 2),
    )
    x = torch.randn(16, 1, 8, 8)
    with torch.no_grad():
        y = black_box(x).argmax(1)

    exp = fit_image(y.numpy(), x.numpy(), epochs=4, batch_size=8, learning_rate=0.05, device="cpu")
    assert exp.modality == "image"
    assert exp.model.classes == 2
    assert exp.model.a.device.type == "cpu"  # device passthrough

    imp = exp.get_explanation(x.numpy())
    assert imp.shape == (16, 8, 8)
    assert np.isfinite(imp).all()

    # ordering is deterministic and covers every pixel exactly once
    order_a = exp.get_order(x)
    order_b = exp.get_order(x)
    assert np.array_equal(order_a, order_b)
    assert np.array_equal(np.sort(order_a.reshape(16, -1), axis=1), np.tile(np.arange(64), (16, 1)))


def test_facade_binary_padded_batch_respects_mask():
    from ruleofthumb import fit_image
    from ruleofthumb.image import pad_images as _pad_images

    torch.manual_seed(1)
    black_box = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 6 * 5, 2),
    )
    small = torch.randn(2, 1, 4, 4)
    big = torch.randn(2, 1, 6, 5)
    padded_probe, _ = _pad_images([small[0], big[0]])
    with torch.no_grad():
        labels = black_box(padded_probe).argmax(1)

    padded, mask = _pad_images([small[0], big[0]])
    exp = fit_image(labels.numpy(), padded.numpy(), mask=mask.numpy(), epochs=4, batch_size=2, learning_rate=0.05)
    imp = exp.get_explanation(padded.numpy(), mask=mask.numpy())
    assert imp.shape == (2, 6, 5)
    assert (imp[0, 4:, :] == 0).all()  # sample 0's bottom rows are padding
    assert (imp != 0).any()


def _write_png(path, height, width, colour=(120, 60, 200)):
    from PIL import Image

    Image.new("RGB", (width, height), colour).save(path)


@pytest.fixture
def png_paths(tmp_path):
    small = tmp_path / "small.png"
    big = tmp_path / "big.png"
    _write_png(small, 2, 3)
    _write_png(big, 4, 4)
    return [str(small), str(big)]


def test_load_images_native_sizes_pad_with_masks(png_paths):
    from ruleofthumb.image import load_images

    out = load_images(png_paths)
    assert out.images.shape == (2, 3, 4, 4)
    assert out.images.dtype == np.float32
    assert out.mask.shape == (2, 4, 4)
    assert out.mask.dtype == bool
    assert int(out.mask[0].sum()) == 2 * 3  # only the small image's real region
    assert out.mask[1].all()
    assert (out.images.transpose(0, 2, 3, 1)[~out.mask] == 0).all()
    assert ((out.images >= 0) & (out.images <= 1)).all()  # [0,1] RGB floats


def test_load_images_fixed_size_resizes_and_crops(png_paths):
    from ruleofthumb.image import load_images

    out = load_images(png_paths, size=(3, 5))
    assert out.images.shape == (2, 3, 3, 5)
    assert out.mask.all()  # uniform size: every pixel is real


def test_load_images_transform_override(png_paths):
    import torch as th

    from ruleofthumb.image import load_images

    def transform(pil_image):
        return th.full((1, 2, 2), 0.5)

    out = load_images(png_paths, transform=transform)
    assert out.images.shape == (2, 1, 2, 2)
    assert (out.images == 0.5).all()
    assert out.mask.all()


def test_fit_routes_image_paths_to_image(png_paths):
    from ruleofthumb import fit

    y = np.array([0, 1])
    exp = fit(y, png_paths, size=(2, 2), epochs=2, batch_size=2, learning_rate=0.05)
    assert exp.modality == "image"
    assert exp.get_explanation(png_paths).shape == (2, 2, 2)


def test_facade_methods_accept_paths(png_paths):
    from ruleofthumb import fit_image

    y = np.array([0, 1])
    exp = fit_image(y, png_paths, size=(2, 2), epochs=2, batch_size=2, learning_rate=0.05)

    imp = exp.get_explanation(png_paths)
    assert imp.shape == (2, 2, 2)
    order = exp.get_order(png_paths)
    assert order.shape == imp.shape
    curve = exp.score_ordering(png_paths, y, order)
    assert curve.ndim == 1 and len(curve) > 0
    preds = exp.predict(png_paths)
    assert preds.shape[0] == 2


def test_paths_with_explicit_mask_rejected(png_paths):
    from ruleofthumb import fit_image

    with pytest.raises(ValueError, match="automatically"):
        fit_image(np.array([0, 1]), png_paths, mask=np.ones((2, 2, 2), dtype=bool))


def test_get_explanation_paths_on_array_fitted_raises(image_data):
    from ruleofthumb import fit_image

    x, y = image_data
    exp = fit_image(y.numpy(), x, epochs=2, batch_size=8, learning_rate=0.05)
    with pytest.raises(ValueError, match="fitted on image arrays"):
        exp.get_explanation(["a.png", "b.png"])
