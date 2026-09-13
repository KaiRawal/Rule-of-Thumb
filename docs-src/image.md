# Images

Pictures in, one importance number per pixel out. Image classifiers
are a case where RoT's frugality really shows — explaining one
decision the classical way can mean querying the model hundreds of
extra times, which you may neither afford nor even be allowed to do.
Start from file paths, and decoding, padding, and masks are all
handled for you:

```python
import ruleofthumb as rot

paths = ["cat.jpg", "dog.jpg"]
exp = rot.fit_image(y_answers, paths)
imp = exp.get_explanation(paths)  # [N, H, W], signed, on the map grid
```

Paths travel through a frozen backbone (`mobilenet_v3_small`) by
default, and that default is worth keeping: raw pixels pool down to
bare ink mass, which caps accuracy on anything beyond the simplest
tasks (see [Limits](capacity.md)). Pass `backbone=None` for raw
pixels, or hand over a torch module for a custom trunk — whichever
you choose is recorded on the explainer and in save files.

If you need custom preprocessing (ImageNet normalisation for a
torchvision black box, say), supply `transform=` — a PIL image to
tensor callable — or inspect the pieces yourself with
`rot.load_images` and `rot.embed_images` directly.

## When you already hold the arrays

Pictures of mixed sizes batch together with `pad_images`:

```python
from ruleofthumb.image import pad_images

x, mask = pad_images(images)  # [N, C, H, W], [N, H, W] True = real
exp = rot.fit_image(y_answers, x.numpy(), mask=mask.numpy())
imp = exp.get_explanation(x.numpy(), mask=mask.numpy())  # [N, H, W]
```

Image orders keep their spatial layout `[N, H, W]` — flat pixel
indices, with `-1` marking filler. One reveal step always covers a
whole **pixel**, its channels travelling together.

## Drawing the answer

```python
fig = plot.saliency(imp[0], image=rgb_array)  # red toward, blue against
```

The gallery is under [Plots](plots.md). For runnable versions, begin
with the gentle [Shapes demo](notebooks/06_shapes_demo.ipynb) —
circles on blank backgrounds, no downloads — then try
[Image demo](notebooks/03_image_quickstart.ipynb) for raw arrays, and
keep [Gaze](notebooks/05_salicon.ipynb) for when you want the
advanced reading.
