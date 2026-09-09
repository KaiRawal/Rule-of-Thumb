# Quickstart

All three modalities on one page. For depth, see the per-modality guides
([Tabular](tabular.md), [Text](text.md), [Image](image.md)) and
[Workflows](workflows.md).

## Tabular

```python
import numpy as np
import ruleofthumb as rot

X_train = np.random.rand(1000, 4).astype(np.float32)
black_box_probs = (X_train[:, 0] > 0.5).astype(np.int64)

exp = rot.fit(y_outputs=black_box_probs, x_inputs=X_train)
importances = exp.get_explanation(X_train)
```

## Text / token embeddings

```python
import numpy as np
import ruleofthumb as rot
from ruleofthumb.text import lengths_to_mask, pad_sequences

sequences = [np.random.rand(t, 384).astype(np.float32) for t in (20, 14, 17, 9)]
x, lengths = pad_sequences(sequences)
labels = np.array([1, 0, 1, 0], dtype=np.int64)
mask = lengths_to_mask(lengths, x.shape[1]).numpy()

exp = rot.fit_text(y_outputs=labels, x_inputs=x.numpy(), mask=mask)
token_importances = exp.get_explanation(x.numpy(), mask=mask)
```

Raw strings are embedded automatically (default
`answerdotai/ModernBERT-base`):

```python
import ruleofthumb as rot

exp = rot.fit_text(y_outputs=labels, x_inputs=["a wonderful film", "terrible pacing"])
token_importances = exp.get_explanation(["a wonderful film", "terrible pacing"])
```

## Images

```python
import numpy as np
import torch
import ruleofthumb as rot
from ruleofthumb.image import pad_images

images = [np.random.rand(3, h, w).astype(np.float32) for h, w in [(32, 32), (28, 40)]]
labels = torch.randint(0, 2, (2,))

x, mask = pad_images(images)
exp = rot.fit_image(y_outputs=labels, x_inputs=x.numpy(), mask=mask.numpy())
imp = exp.get_explanation(x.numpy(), mask=mask.numpy())
```

Starting from image files? Paths embed through a frozen backbone by
default (`mobilenet_v3_small`) — raw pixels pool to ink mass and cap
fidelity on focal tasks, so prefer maps (pass `backbone=None` for pixels):

```python
import ruleofthumb as rot

paths = ["cat.jpg", "dog.jpg"]
exp = rot.fit_image(y_outputs=labels, x_inputs=paths)
imp = exp.get_explanation(paths)   # signed, shape [N, h, w] on the map grid
```

Image file paths work directly too — see [Image](image.md) and the
`notebooks/03_image_quickstart.ipynb` notebook.

## Next steps

- [API reference](api.md) for every public entry point.
- Executed *Notebooks* for runnable hello-worlds.
- [Capacity](capacity.md) before trusting an explanation.
