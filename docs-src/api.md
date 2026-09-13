# API reference

Every public function and class, rendered from docstrings. New here?
Read [Core ideas](concepts.md) first, then pick your task below.

## Fit: make an explainer

| Symbol | One line |
|---|---|
| `fit` | Detect the modality from the input shape and fit |
| `fit_tabular` | Fit on tables `[N, columns]` |
| `fit_text` | Fit on sentences (strings or `[N, tokens, dim]` + mask) |
| `fit_image` | Fit on pictures (paths or `[N, C, H, W]` + mask) |
| `Explainer` | The fitted object both factories return |
| `load_explainer` | Reload a saved explainer without refitting |

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

## Explain: ask it questions

The `Explainer` methods above are the whole asking surface:

- `get_explanation(inputs)` — signed importances: tables `[N, D]`,
  sentences `[N, T]`, pictures `[N, H, W]` (one set per answer when
  there are more than two). Filler scores exactly zero.
- `get_order(inputs)` — most-important-first ranking per row; filler
  ranks last as `-1`.
- `ordered_predict(inputs, order)` — answers while uncovering in rank
  order. `score_ordering(inputs, answers, order)` — accuracy per step.
- `score` / `predict` — the stand-in's own answers.

## Tune and ingest

| Symbol | One line |
|---|---|
| `autotune` / `AutotuneResult` | Search settings, refit the winner on all data |
| `embed_texts` / `TextEmbeddings` | Strings → embeddings, mask, decoded tokens |
| `load_images` / `ImageBatch` | Files → pixels plus validity mask |
| `embed_images` / `ImageEmbeddings` | Files → backbone maps plus mask |

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

## Pad: batch the ragged

| Symbol | One line |
|---|---|
| `pad_sequences` | Ragged embedding lists → one batch plus lengths |
| `lengths_to_mask` | Lengths → boolean mask (`True` = real word) |
| `pad_images` | Mixed-size pictures → one batch plus mask |

```{eval-rst}
.. autofunction:: ruleofthumb.pad_sequences
```

```{eval-rst}
.. autofunction:: ruleofthumb.lengths_to_mask
```

```{eval-rst}
.. autofunction:: ruleofthumb.pad_images
```

## Raw models and plots

`core.RoT`, `text.RoTText`, and `image.RoTImage` are the un-facaded
models for full manual control — reach for them only when the
factories get in your way.

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

`ruleofthumb.plot` draws everything (see [Plots](plots.md)):

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
