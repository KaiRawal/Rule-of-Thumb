# Images

Pictures in, one importance number per pixel out. Start from file
paths — decoding, padding, and masks are handled for you:

```python
import ruleofthumb as rot

paths = ["cat.jpg", "dog.jpg"]
exp = rot.fit_image(y_answers, paths)
imp = exp.get_explanation(paths)  # [N, H, W], signed, on the map grid
```

Paths are embedded through a frozen backbone (`mobilenet_v3_small`)
by default. Prefer that: raw pixels pool down to ink mass and cap
accuracy on anything but the simplest tasks (see [Limits](capacity.md)).
Pass `backbone=None` for raw pixels, or a torch module for a custom
trunk. The choice is recorded on the explainer and in save files.

Need custom preprocessing (e.g. ImageNet normalisation)? Supply
`transform=` (a PIL image → tensor callable), or inspect the pieces
with `rot.load_images` / `rot.embed_images` directly.

## Power mode: arrays and masks

Mixed-size pictures batch with `pad_images`:

```python
from ruleofthumb.image import pad_images

x, mask = pad_images(images)  # [N, C, H, W], [N, H, W] True = real
exp = rot.fit_image(y_answers, x.numpy(), mask=mask.numpy())
imp = exp.get_explanation(x.numpy(), mask=mask.numpy())  # [N, H, W]
```

Image orders keep the spatial layout `[N, H, W]` (flat pixel indices,
`-1` = filler). One reveal step covers a whole **pixel** (its channels
go together).

## Draw it

```python
fig = plot.saliency(imp[0], image=rgb_array)  # red toward, blue against
```

More: [Plots](plots.md). The gentle hello-world is the
[Shapes demo](notebooks/06_shapes_demo.ipynb) — circles on blank
backgrounds, no downloads. [Image demo](notebooks/03_image_quickstart.ipynb)
covers raw arrays; [Gaze](notebooks/05_salicon.ipynb) is advanced reading.
