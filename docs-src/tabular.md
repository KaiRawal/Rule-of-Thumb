# Tabular guide

Fits a transparent surrogate to a black box on vector inputs, returning one
signed importance weight per feature. Positive means evidence toward class 1
(binary); for `n_classes > 2` the output is per-class.

## Minimal fit

```python
import numpy as np
import ruleofthumb

X_train = np.random.rand(1000, 4).astype(np.float32)
black_box_probs = (X_train[:, 0] > 0.5).astype(np.int64)  # e.g. model.predict(X_train)

exp = ruleofthumb.fit(y_outputs=black_box_probs, x_inputs=X_train)   # or fit_tabular(...)
importances = exp.get_explanation(X_train)  # signed, shape [N, d]
```

`fit` auto-detects 2-D inputs as tabular; `fit_tabular` is the explicit
equivalent. Both accept `n_classes=`, `seed=`, `pretrain_epochs=`,
`weight_decay=` and `device=`; see [API](api.md#ruleofthumb.fit_tabular).

## Reveal pipeline

`get_order` ranks features most-important-first per sample;
`ordered_predict` re-scores the black-box labels as features are revealed;
`score_ordering` summarises fidelity along the curve (default: per-step
accuracy; `return_confusion=True` gives per-step K×K counts). Tabular is
unaffected by the `granularity=` setting (one step per feature; see
[Workflows](workflows.md#reveal-curves)).

## Worked notebook

The executed hello-world is
`notebooks/01_tabular_quickstart.ipynb` (generated copy of
`examples/01_tabular_quickstart.ipynb`, re-executed on every site build):
a toy black box on synthetic data, surrogate fit, and importance inspection.

## Next steps

- Cross-cutting topics (tuning, persistence, non-linear shapes, plotting,
  devices): [Workflows](workflows.md).
- Raw model control (`RoT`, `score`, `importance`): [API](api.md#ruleofthumb.core.RoT).
- Capacity limits are modest here — the pooled-capacity caveat in
  [Capacity](capacity.md) applies to text and images.
