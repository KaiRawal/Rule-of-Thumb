# Quickstart

Your first explanation in five minutes, using a table of numbers —
the same shape as any scikit-learn dataset. Everything below runs on
CPU in seconds.

## 1. Make a toy black box

A "black box" is just something that turns inputs into answers and
won't tell you how. Here the box calls a row class 1 when its first
two columns sum past 1:

```python
import numpy as np
import ruleofthumb as rot

rng = np.random.RandomState(0)
X = rng.rand(1000, 4).astype(np.float32)
y_answers = ((X[:, 0] + X[:, 1]) > 1.0).astype(np.int64)  # e.g. model.predict(X)
```

## 2. Fit the stand-in

`fit_tabular` learns a simple model that copies those answers:

```python
exp = rot.fit_tabular(y_answers, X, epochs=30, seed=0)
print(f"agreement: {exp.train_agreement_:.2f}")  # stand-in vs box, want ~1.0
```

`train_agreement_` is the fraction of answers the stand-in gets right.
Near 1 means the explanation below is worth reading; far below means
stop and check [Limits](capacity.md) first.

## 3. Read the explanation

One signed number per column, per row. Positive pushes *toward* the
predicted answer, negative pushes *away*:

```python
imp = exp.get_explanation(X[:5])  # shape [5, 4]
print(imp[0])  # e.g. [0.9, 0.7, 0.0, -0.0]: columns 0 and 1 did the work
```

## 4. Rank and reveal

`get_order` sorts columns most-important-first per row; `score_ordering`
uncovers them in that order and re-checks the answer at each step:

```python
import torch

points = torch.from_numpy(X[:200])
labels = torch.from_numpy(y_answers[:200])
order = exp.get_order(points)
curve = exp.score_ordering(points, labels, order)
print(curve)  # accuracy after 0, 1, 2, ... columns revealed
```

A good order reaches full accuracy after one or two columns. Plot it
with `plot.reveal` (see [Reveal curves](reveal.md)):

```{image} _static/figures/reveal.light.png
:class: only-light
:alt: Accuracy climbing to 1 after one revealed input for the learned order, slowly for random
```

```{image} _static/figures/reveal.dark.png
:class: only-dark
:alt: Accuracy climbing to 1 after one revealed input for the learned order, slowly for random
```

## 5. Draw it and keep it

```python
from ruleofthumb import plot

fig = plot.waterfall(exp, X[:1], feature_names=["size", "age", "zip", "color"])
fig.savefig("waterfall.png")

exp.save("explainer.rotx")
loaded = rot.load_explainer("explainer.rotx")
```

`plot.waterfall` shows one answer as stacked contributions; every plot
function returns a `Figure` and never shows it for you. See
[Plots](plots.md) for the full gallery.

## Next steps

- Sentences instead of tables? [Text](text.md).
- Pictures instead of tables? [Images](image.md) and the runnable
  [Shapes demo](notebooks/06_shapes_demo.ipynb).
- The ideas behind the calls? [Core ideas](concepts.md).
- Every function documented? [API reference](api.md).
