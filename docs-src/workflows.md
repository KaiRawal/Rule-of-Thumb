# Cross-modality workflows

Topics shared by all three modalities: reveal curves, tuning, persistence,
non-linear shapes, plotting and devices.

(reveal-curves)=

## Reveal curves

`get_order` → `ordered_predict` → `score_ordering` simulates revealing
inputs most-important-first and scores prediction fidelity along the curve:

- Reveal units are features (tabular), tokens (text) or pixels (images) by
  default; embedding dims / channels aggregate via abs-sum.
- The pipeline is mask-aware: padded positions rank last (reported as
  `-1`); by default curves stop after each sample's real features are
  exhausted. `include_padded=True` retains the full rectangular curve.
- `granularity="element"` restores per-feature-element curves; the value
  must match how the order was produced (no auto-detection). Tabular is
  unaffected.
- `score_ordering` defaults to per-step accuracy (any class count); pass
  `return_confusion=True` for per-step K×K confusion counts (rows = true
  label, columns = predicted class), or `metric=` for a custom callable
  over binary counts `(tp, fp, fn, tn)` — use custom metrics only where
  that view is meaningful.
- Argument order is `score_ordering(points, labels, order)`; swapping the
  first two raises a `ValueError` naming the swap instead of failing deep
  inside torch.

## Automatic hyperparameter tuning

`rot.autotune` searches `learning_rate` / `batch_size` / `epochs`
/ `dropout_rate` / `weight_decay` with a seeded validation split, scores
candidates by held-out reveal fidelity, and returns the winner refit on
all data:

```python
result = rot.autotune(y_outputs=labels, x_inputs=x, search="random",
                              n_candidates=8, seed=0)
result.explainer    # best config refit on all data — use like any explainer
result.best_params  # winning hyperparameters
result.trials       # every candidate with its validation score, best-first
```

`search="grid"` enumerates a `space=` dict exhaustively; `space=` accepts
any subset of the defaults. Works for all three modalities, including raw
strings and image paths.

`n_classes` is inferred from the labels (override explicitly when a split
might miss a class). Any other factory keyword (`nonlinear`,
`l1_penalty`, `dropout_rate`, `mask`, ...) is forwarded to both the
candidate fits and the final refit; a per-sample `mask` is split alongside
the data:

```python
result = rot.autotune(y_outputs=labels, x_inputs=x, search="grid",
                              space={"epochs": [100, 300]},
                              nonlinear="hinge", mask=mask, seed=0)
```

## Saving and loading

Fitted explainers round-trip through `Explainer.save` /
`load_explainer` (weights + configuration only — no refitting, no pickled
classes):

```python
exp.save("explainer.rotx")
loaded = rot.load_explainer("explainer.rotx", device="cpu")
np.allclose(exp.get_explanation(x), loaded.get_explanation(x))  # identical
```

Native string / file-path ingestion is not persisted: a reloaded explainer
consumes numeric arrays (refit from strings/paths to restore it).

## Non-linear additive explanations

By default every surrogate is linear. Pass `nonlinear=` to any factory
(all modalities, binary and multiclass) to learn a shared elementwise
response `s` inside `imp[k,i] = a[k,i]·(s(x[i]) + b[k,i])`:

```python
exp = rot.fit_tabular(y, x, nonlinear="rbf")            # Gaussian bumps
exp = rot.fit_text(y, texts, nonlinear="hinge")         # SELU hinges
exp = rot.fit_image(y, paths, nonlinear={"type": "rbf", "n_bases": 32})
```

Both responses are residual and zero-initialised, so an unfitted
non-linear model is exactly the linear one. Explanations stay exactly
additive, so plotting, reveal curves and persistence work unchanged; the
configuration round-trips through save files. Omitting `nonlinear` keeps
the plain linear model.

(plotting)=

## Plotting

`rot.plot` renders every modality (**red = evidence toward the
explained class, blue = against**); everything returns a Figure, nothing
auto-shows:

```python
import ruleofthumb.plot as plot

# Tabular — SHAP's signature plots via the shap package:
plot.waterfall(exp, x[:1], feature_names=names)   # also: force, decision
plot.bar(exp, x[:50], feature_names=names)        # also: beeswarm (batch-level)

# Text — token highlighting plus static export and word clouds:
plot.text_html(imp[0], out.tokens[0])             # IPython-aware HTML
plot.text_matplotlib(imp[0], out.tokens[0])       # static matplotlib export
plot.word_clouds(imp, out.tokens)                 # pos/neg/combined clouds

# Images — saliency overlay over optional RGB input:
plot.saliency(imp_map, image=rgb_array)
```

**Baseline semantics (SHAP → RoT).** SHAP decomposes `f(x) = φ₀ + Σφᵢ`
with `φ₀ = E[f(X)]`. RoT's surrogate is additive by construction:
`s_k(x) = g_k + Σ_d a_kd·(x_d + b_kd)` and `get_explanation` returns
exactly the per-feature terms — so the SHAP base value maps to the
**class bias `g_k`**, and `f(x)` maps to the **surrogate score** (RoT
explains its own surrogate of the black box, not the black box
directly). Text scores are length-normalised means over tokens, so token
importances do not sum to the score.

## Devices

All three RoT models and the explainer factories accept `device=`
(`None` auto-detects CUDA → MPS → CPU). Fit and inference move inputs
automatically; every inference entry point (`score`, `predict`,
`ordered_predict`, `score_ordering`, `get_order`, `get_explanation`)
returns host-side results (CPU tensors or numpy arrays) regardless of
`device`. Only the training internals (`importance`,
`stochastic_importance`, `training_loop`) stay on the model's device.

```python
exp = rot.fit(y_outputs=labels, x_inputs=X, device="cuda")  # or "mps", "cpu", ...
```
