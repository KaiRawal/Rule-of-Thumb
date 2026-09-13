# Quickstart

Let's fit your first explanation. It takes about five minutes, uses a
small table of numbers — the same shape as any scikit-learn dataset —
and everything below runs on CPU in seconds.

## Preparing a toy black box

A "black box" is simply something that turns inputs into answers
without telling you how. Ours will be deliberately transparent: a row
counts as class 1 whenever its first two columns sum past 1. (In real
life you would put your model's own predictions here — the code does
not care where the answers came from.)

```python
import numpy as np
import ruleofthumb as rot

rng = np.random.RandomState(0)
X = rng.rand(1000, 4).astype(np.float32)
y_answers = ((X[:, 0] + X[:, 1]) > 1.0).astype(np.int64)  # e.g. model.predict(X)
```

## Fitting the stand-in

`fit_tabular` learns a small model whose only job is to copy those
answers:

```python
exp = rot.fit_tabular(y_answers, X, epochs=30, seed=0)
print(f"agreement: {exp.train_agreement_:.2f}")  # stand-in vs box, want ~1.0
```

That agreement score is the fraction of answers the stand-in gets
right. Close to 1 means the explanation below is worth reading; if it
ever lands far below, stop and visit [Limits](capacity.md) before
going further.

## Reading the explanation

You get one signed number per column, per row. Positive numbers
pushed *toward* the predicted answer, negative ones pushed *away*:

```python
imp = exp.get_explanation(X[:5])  # shape [5, 4]
print(imp[0])  # e.g. [0.9, 0.7, 0.0, -0.0]: columns 0 and 1 did the work
```

Columns 2 and 3 sit near zero, which is exactly right — the box never
looked at them. That moment, where the numbers confirm something you
already knew, is a good way to calibrate your trust before moving on
to models you cannot see inside.

## Ranking inputs and checking the ranking

`get_order` sorts each row's columns most-important-first, and
`score_ordering` uncovers them in that order, re-checking the answer
at every step:

```python
import torch

points = torch.from_numpy(X[:200])
labels = torch.from_numpy(y_answers[:200])
order = exp.get_order(points)
curve = exp.score_ordering(points, labels, order)
print(curve)  # accuracy after 0, 1, 2, ... columns revealed
```

A ranking you can believe in reaches full accuracy after one or two
columns. `plot.reveal` draws that story for you (see
[Checking the ranking by revealing less](reveal.md)):

```{image} _static/figures/reveal.light.png
:class: only-light
:alt: Accuracy climbing to 1 after one revealed input for the learned order, slowly for random
```

```{image} _static/figures/reveal.dark.png
:class: only-dark
:alt: Accuracy climbing to 1 after one revealed input for the learned order, slowly for random
```

## Drawing it and keeping it

```python
from ruleofthumb import plot

fig = plot.waterfall(exp, X[:1], feature_names=["size", "age", "zip", "color"])
fig.savefig("waterfall.png")

exp.save("explainer.rotx")
loaded = rot.load_explainer("explainer.rotx")
```

`plot.waterfall` renders a single answer as stacked contributions,
and like every plotting function it hands you a `Figure` without
displaying anything — saving and showing stay your decision. The
full gallery lives under [Plots](plots.md).

## Where to go from here

- Working with sentences rather than tables? Head to [Text](text.md).
- Working with pictures? [Images](image.md) awaits, with the runnable
  [Shapes demo](notebooks/06_shapes_demo.ipynb) alongside it.
- Curious about the ideas behind these calls? [Core ideas](concepts.md)
  tells that story from the beginning.
- After something specific? The [API reference](api.md) documents
  every public function.
