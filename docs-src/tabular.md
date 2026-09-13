# Tabular

Tables of numbers: `X` is `[rows, columns]`, `y` is one integer answer
per row. One importance number per column.

## Try it

```python
import numpy as np
import ruleofthumb as rot

X = np.random.RandomState(0).rand(1000, 4).astype(np.float32)
y_answers = ((X[:, 0] + X[:, 1]) > 1.0).astype(np.int64)

exp = rot.fit_tabular(y_answers, X, epochs=30, seed=0)
imp = exp.get_explanation(X)  # [1000, 4], signed
```

`fit` with a 2-D input does the same thing (it detects tables
automatically); `fit_tabular` says it explicitly.

## Shapes and meanings

- Binary (two answers): `get_explanation` returns `[N, D]` — the
  contributions toward answer 1. Positive pushes toward 1, negative
  toward 0.
- More answers (`n_classes=`): returns `[N, K, D]` — one set per
  answer, never collapsed.
- Tabular takes no mask: every cell is real data.

## Rank, reveal, draw

```python
order = exp.get_order(X_torch)                    # [N, D], best first
curve = exp.score_ordering(X_torch, y_torch, order)  # accuracy per step
fig = plot.waterfall(exp, X[:1], feature_names=names)  # one answer
fig = plot.bar(exp, X[:50], feature_names=names)       # whole batch
```

Details: [Reveal curves](reveal.md), [Plots](plots.md).
The executed hello-world is
[Tabular demo](notebooks/01_tabular_quickstart.ipynb).

## Next steps

- Better fits and keeping them: [Tuning and saving](workflows.md).
- What to check before trusting: [Limits](capacity.md) (tables are
  the easy case — the caveats bite text and images).
