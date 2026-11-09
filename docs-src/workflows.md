# Tuning and saving

Once the basics feel comfortable, two things tend to come up: "could
this fit be better?" and "how do I keep it?" This page answers both,
then gestures at going beyond straight lines and at choosing hardware.

## Letting the package search for you

`rot.autotune` tries a spread of learning rates, batch sizes, epochs,
and weight-decay settings, keeps whichever scores best on held-out
agreement, and refits the winner on all of your data. Scoring stays
deliberately plain — just surrogate-vs-label accuracy, no reveal
machinery — unless you explicitly ask for `scoring="reveal"`, which
scores candidates by final-step reveal accuracy instead. Dropout stays
fixed at 0.5 — the method needs every partial observation equally
likely:

```python
result = rot.autotune(y_answers, X, search="random", n_candidates=8, seed=0)
result.explainer    # use like any explainer
result.best_params  # winning settings
result.trials       # every try, best first
```

Prefer an exhaustive sweep? `search="grid"` with something like
`space={"epochs": [100, 300]}` tries every combination instead. Any
other keyword (`mask=`, `nonlinear=`, …) rides along into each attempt
and the final refit.

Each kind of input has its own default spread to search
(`ruleofthumb.tune.DEFAULT_SPACES`): text needs far smaller learning
rates than tables. With four or more candidates, `rot.autotune` trains
each one for about a third of its epochs first and finishes only the
better half; a candidate that runs out of memory is skipped rather than
ending the search.

### Every fit runs `rot.autotune` by default

`fit`, `fit_tabular`, `fit_text` and `fit_image` call `rot.autotune`
for you (`tune=True` is the default) and return the refit winner, with
the search's outcome attached:

```python
exp = rot.fit_tabular(y_answers, X, seed=0)
exp.best_params_   # same as result.best_params
exp.best_score_    # its held-out agreement
```

A dict passes options through to `rot.autotune`:
`tune={"search": "grid", "space": {"epochs": [100, 300]}}` accepts
`search`, `space`, `n_candidates`, `scoring` and `validation_split`.
Want exactly your own settings, with no search? Pass `tune=None` —
also the fast path for tests and quick experiments. Its defaults suit
tiny inputs — do scale `epochs` and `batch_size` (16–64 is a sensible
neighbourhood) to the size of your own data:

```python
exp = rot.fit_tabular(y_answers, X, epochs=30, batch_size=500, learning_rate=0.05,
                      seed=0, tune=None)
```

## Stopping once the fit stops improving

Rather than guess `epochs`, let training stop itself: with
`patience=3`, agreement is checked every `check_every=10` epochs and
training ends after three checks without a gain of at least
`min_delta=1e-3`. `epochs` becomes the upper limit. It is off by
default, and works with or without the search (each candidate and the
refit stop early).

```python
exp = rot.fit_tabular(y_answers, X, epochs=2000, patience=3, seed=0, tune=None)
```

## Checking a fit on new data

`exp.evaluate(X_new, y_new)` reports how often the stand-in agrees with
the black box on data you choose (ideally data it never trained on),
warning below 0.75. `reveal=True` adds the reveal-curve area against a
random order (`reveal_gap`); `rationales=` (a boolean mask of the units
a person marked as important) adds how well the importances find them
(`rationale_auroc`). It never refits.

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
