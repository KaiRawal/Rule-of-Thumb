# Tabular

Tables of numbers are the gentlest place to begin: `X` holds
`[rows, columns]`, `y` holds one integer answer per row, and you get
back one importance number per column — no masks, no padding, no
embeddings to think about.

## Seeing it on a small table

```python
import numpy as np
import ruleofthumb as rot

X = np.random.RandomState(0).rand(1000, 4).astype(np.float32)
y_answers = ((X[:, 0] + X[:, 1]) > 1.0).astype(np.int64)

exp = rot.fit_tabular(y_answers, X, epochs=30, seed=0)
imp = exp.get_explanation(X)  # [1000, 4], signed
```

Calling `fit` with a 2-D input does exactly the same thing — it
recognises a table automatically — while `fit_tabular` states your
intention out loud. Either is fine; pick the one that reads better
in your code.

## What the shapes mean

- With two possible answers, `get_explanation` returns `[N, D]`:
  each row's contributions toward answer 1. Positive values pushed
  toward 1, negative ones toward 0.
- With more answers (`n_classes=`), you get `[N, K, D]` instead — a
  full set of contributions per answer, never collapsed into one.
- Tables take no mask at all: every cell is genuine data, so there is
  nothing to mark as filler.

## From numbers to a ranking you can check

```python
order = exp.get_order(X_torch)                    # [N, D], best first
curve = exp.score_ordering(X_torch, y_torch, order)  # accuracy per step
fig = plot.waterfall(exp, X[:1], feature_names=names)  # one answer
fig = plot.bar(exp, X[:50], feature_names=names)       # whole batch
```

The ranking and the reveal curve are covered properly under
[Checking the ranking by revealing less](reveal.md), and the drawings
under [Plots](plots.md). If you would like to watch the whole
pipeline run, the [Tabular demo](notebooks/01_tabular_quickstart.ipynb)
executes it end to end.

## Where to go from here

- Fitting better and keeping the result: [Tuning and saving](workflows.md).
- What to verify before trusting: [Limits](capacity.md) — though
  tables are honestly the easy case, and most of the caveats there
  concern text and images.
