# API reference

This is the complete contract for every public function and class,
rendered from the docstrings. If you are new here, you will enjoy it
far more after [Core ideas](concepts.md) — then come back and pick
the task that matches yours below.

## Fitting an explainer

| Symbol | What it does for you |
|---|---|
| `fit` | Reads the shape of your inputs, picks the modality, and fits |
| `fit_tabular` | Fits on tables shaped `[N, columns]` |
| `fit_text` | Fits on sentences — raw strings or `[N, tokens, dim]` plus a mask |
| `fit_image` | Fits on pictures — file paths or `[N, C, H, W]` plus a mask |
| `Explainer` | The fitted object every factory returns, and everything you ask afterwards |
| `load_explainer` | Brings a saved explainer back without refitting |

```{eval-rst}
.. autofunction:: ruleofthumb.fit
```

```{eval-rst}
.. autofunction:: ruleofthumb.fit_tabular
```

```{eval-rst}
.. autofunction:: ruleofthumb.fit_text
```

```{eval-rst}
.. autofunction:: ruleofthumb.fit_image
```

```{eval-rst}
.. autoclass:: ruleofthumb.Explainer
   :members: get_explanation, get_order, ordered_predict, score_ordering, score, predict, save
```

```{eval-rst}
.. autofunction:: ruleofthumb.load_explainer
```

## Asking a fitted explainer

The `Explainer` methods above are the entire question surface — once
fitted, everything you can ask lives here:

- `get_explanation(inputs)` returns the signed importances: tables
  `[N, D]`, sentences `[N, T]`, pictures `[N, H, W]` (a full set per
  answer when there are more than two). Filler always scores exactly
  zero.
- `get_order(inputs)` ranks each row most-important-first, with filler
  trailing last as `-1`.
- `ordered_predict(inputs, order)` replays the answers while
  uncovering in rank order, and `score_ordering(inputs, answers,
  order)` turns that replay into accuracy per step.
- `score` and `predict` simply report the stand-in's own answers.

## Tuning, and getting data in

| Symbol | What it does for you |
|---|---|
| `autotune` / `AutotuneResult` | Searches the settings for you, then refits the winner on all data |
| `embed_texts` / `TextEmbeddings` | Turns strings into embeddings, a mask, and decoded tokens |
| `load_images` / `ImageBatch` | Turns files into pixels alongside a validity mask |
| `embed_images` / `ImageEmbeddings` | Turns files into backbone maps alongside a mask |

```{eval-rst}
.. autofunction:: ruleofthumb.autotune
```

```{eval-rst}
.. autoclass:: ruleofthumb.AutotuneResult
   :members:
```

```{eval-rst}
.. autofunction:: ruleofthumb.embed_texts
```

```{eval-rst}
.. autoclass:: ruleofthumb.TextEmbeddings
   :members:
```

```{eval-rst}
.. autodata:: ruleofthumb.DEFAULT_TEXT_MODEL
```

```{eval-rst}
.. autodata:: ruleofthumb.DEFAULT_TEXT_REVISION
```

```{eval-rst}
.. autofunction:: ruleofthumb.load_images
```

```{eval-rst}
.. autoclass:: ruleofthumb.ImageBatch
   :members:
```

```{eval-rst}
.. autofunction:: ruleofthumb.embed_images
```

```{eval-rst}
.. autoclass:: ruleofthumb.ImageEmbeddings
   :members:
```

```{eval-rst}
.. autodata:: ruleofthumb.DEFAULT_IMAGE_MODEL
```

## Batching the ragged

Sentences and pictures arrive in all shapes, but models eat
rectangles — these helpers bridge that gap:

| Symbol | What it does for you |
|---|---|
| `pad_sequences` | Gathers ragged embedding lists into one batch, plus their lengths |
| `lengths_to_mask` | Turns those lengths into a boolean mask (`True` marks a real word) |
| `pad_images` | Gathers mixed-size pictures into one batch, plus a mask |

```{eval-rst}
.. autofunction:: ruleofthumb.pad_sequences
```

```{eval-rst}
.. autofunction:: ruleofthumb.lengths_to_mask
```

```{eval-rst}
.. autofunction:: ruleofthumb.pad_images
```

## Raw models, and the plots that draw them

Under the factories sit `core.RoT`, `text.RoTText`, and
`image.RoTImage` — the unfacaded models with full manual control.
Reach for them only when the factories start getting in your way.

```{eval-rst}
.. autoclass:: ruleofthumb.core.RoT
   :members:
```

```{eval-rst}
.. autoclass:: ruleofthumb.text.RoTText
   :members:
```

```{eval-rst}
.. autoclass:: ruleofthumb.image.RoTImage
   :members:
```

`ruleofthumb.plot` draws everything — the friendlier tour is under [Plots](plots.md):

```{eval-rst}
.. automodule:: ruleofthumb.plot
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.waterfall
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.force
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.decision
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.bar
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.beeswarm
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.text_html
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.text_matplotlib
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.saliency
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.reveal
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.word_clouds
```
