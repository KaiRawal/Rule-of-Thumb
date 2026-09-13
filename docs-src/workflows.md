# Tuning and saving

Once the basics feel comfortable, two things tend to come up: "could
this fit be better?" and "how do I keep it?" This page answers both,
then gestures at going beyond straight lines and at choosing hardware.

## Letting the package search for you

`rot.autotune` tries a spread of learning rates, batch sizes, epochs,
and weight-decay settings, keeps whichever scores best on held-out reveal
fidelity, and refits the winner on all of your data. Dropout stays fixed
at 0.5 — the method needs every partial observation equally likely:

```python
result = rot.autotune(y_answers, X, search="random", n_candidates=8, seed=0)
result.explainer    # use like any explainer
result.best_params  # winning settings
result.trials       # every try, best first
```

Prefer an exhaustive sweep? `search="grid"` with something like
`space={"epochs": [100, 300]}` tries every combination instead. Any
other keyword (`mask=`, `nonlinear=`, …) rides along into each attempt
and the final refit. The defaults suit tiny inputs — do scale `epochs`
and `batch_size` (16–64 is a sensible neighbourhood) to the size of
your own data.

## Saving your work so you don't refit

```python
exp.save("explainer.rotx")
loaded = rot.load_explainer("explainer.rotx", device="cpu")
```

A reloaded explainer answers identically, with two honest caveats.
Raw-string and file-path ingestion is not persisted, so it consumes
numeric arrays until you refit from strings or paths to restore the
convenience. And `train_agreement_` comes back as `None` — that
tripwire describes a moment in training, not something worth saving.

## When straight lines are not enough

Every stand-in starts linear, which covers a surprising amount of
ground. When you suspect an input matters in the middle of its range
but not at the ends, pass `nonlinear=` to any factory, for any
modality:

```python
exp = rot.fit_tabular(y, X, nonlinear="rbf")
exp = rot.fit_text(y, texts, nonlinear="hinge")
```

Explanations stay exactly additive, so ranking, reveal curves, plots,
and saving all carry on working unchanged. Leave the argument out
and you keep the plain linear model — a perfectly respectable
default.

## Choosing hardware

Factories and models all accept `device=`, where `None` quietly picks
CUDA, then MPS, then CPU, in that order. Training moves your inputs
for you, and every answering method (`predict`, `get_explanation`,
`get_order`, `ordered_predict`, `score_ordering`) returns host-side
NumPy or CPU results regardless of where the fitting happened:

```python
exp = rot.fit(y_answers, X, device="cuda")
```
