# Image guide

Inputs are `(N, channels, height, width)` tensors; importance is shared
across spatial locations (see [Capacity](capacity.md) for what that
implies). Mixed-size batches are supported two ways.

## Padded batches (explainer facade)

```python
import numpy as np
import torch
import ruleofthumb
from ruleofthumb.image import pad_images

images = [np.random.rand(3, h, w).astype(np.float32) for h, w in [(32, 32), (28, 40)]]
labels = torch.randint(0, 2, (2,))

x, mask = pad_images(images)                          # x: (2, 3, 32, 40); mask: (2, 32, 40)
exp = ruleofthumb.fit_image(y_outputs=labels, x_inputs=x.numpy(), mask=mask.numpy())
imp = exp.get_explanation(x.numpy(), mask=mask.numpy())  # signed, shape [N, H, W]
```

## Per-sample loop (raw model)

Weights are size-agnostic, so unpadded samples can be handled one at a time
with no mask:

```python
from ruleofthumb.image import RoTImage

model = RoTImage(classes=2, sample_shape=(3,))
model.fit(torch.from_numpy(x), labels, epochs=50, batch_size=2, lr=0.01, mask=mask)
for img in images:
    single_imp = model.importance(torch.from_numpy(img[None]))
```

## Image files

Pass paths straight in — `fit_image` decodes them (RGB, `[0, 1]` floats),
derives validity masks automatically, and every explainer method accepts
the same paths back:

```python
import ruleofthumb

paths = ["cat.jpg", "dog.jpg"]
exp = ruleofthumb.fit_image(y_outputs=labels, x_inputs=paths)               # native sizes, padded
exp = ruleofthumb.fit_image(y_outputs=labels, x_inputs=paths, size=(64, 64))  # resize + centre-crop
imp = exp.get_explanation(paths)   # signed, shape [N, H, W]
```

Need custom preprocessing (e.g. ImageNet normalisation for a torchvision
black box)? Supply `transform=` (a PIL Image → tensor callable), or use
`ruleofthumb.load_images(paths, ...)` directly to inspect `.images` /
`.mask`.

## Reveal pipeline

One reveal step covers a whole **pixel** (channels revealed together);
`granularity="element"` restores per-element curves. Details:
{ref}`Workflows: reveal curves <reveal-curves>`.

## Worked notebook

The executed hello-world is
`notebooks/03_image_quickstart.ipynb` (generated copy of
`examples/03_image_quickstart.ipynb`, re-executed on every site build).

## Next steps

- Saliency overlays: {ref}`Workflows: plotting <plotting>`.
- Prefer rich channel representations (e.g. MobileNet feature maps) —
  [Capacity](capacity.md) shows why raw pixels cap fidelity.
