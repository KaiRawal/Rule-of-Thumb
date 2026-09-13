# Tuning and saving

## Better fits automatically

`rot.autotune` tries learning rates, batch sizes, epochs, and dropout,
keeps the setting with the best held-out reveal fidelity, and refits
it on all your data:

```python
result = rot.autotune(y_answers, X, search="random", n_candidates=8, seed=0)
result.explainer    # use like any explainer
result.best_params  # winning settings
result.trials       # every try, best first
```

`search="grid"` with `space={"epochs": [100, 300]}` tries everything
in the grid instead. Any other keyword (`mask=`, `nonlinear=`, …) is
passed through to each try and the final refit. The defaults suit tiny
inputs — scale `epochs` / `batch_size` (e.g. 16–64) to your data size.

## Keep a fitted explainer

```python
exp.save("explainer.rotx")
loaded = rot.load_explainer("explainer.rotx", device="cpu")
```

Reloaded explainers answer identically but consume numeric arrays:
raw-string / file-path ingestion is not persisted, so refit from
strings/paths to restore it. Reloaded explainers also report
`train_agreement_ = None` (that tripwire is transient, not saved).

## Beyond straight lines

Every stand-in is linear by default. Pass `nonlinear=` to any factory
(all modalities) to learn a shared per-element curve on top — useful
when an input helps in the middle of its range but not at the ends:

```python
exp = rot.fit_tabular(y, X, nonlinear="rbf")
exp = rot.fit_text(y, texts, nonlinear="hinge")
```

Explanations stay exactly additive, so ranking, reveal curves, plots,
and saving all work unchanged. Omitting `nonlinear` keeps the plain
linear model.

## Devices

Factories and models accept `device=` (`None` picks CUDA → MPS → CPU
automatically). Training moves inputs for you; every answering method
(`predict`, `get_explanation`, `get_order`, `ordered_predict`,
`score_ordering`) returns host-side NumPy/CPU results regardless.

```python
exp = rot.fit(y_answers, X, device="cuda")
```
