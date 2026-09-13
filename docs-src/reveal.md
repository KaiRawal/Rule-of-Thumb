# Reveal curves

The payoff for the whole package: a good ranking keeps the black
box's answer after revealing very little. Three calls:

```python
order = exp.get_order(inputs)                          # best first
preds = exp.ordered_predict(inputs, order)             # answers per step
curve = exp.score_ordering(inputs, answers, order)     # accuracy per step
```

`curve[k]` is the accuracy after uncovering the top-`k` pieces
(columns, words, or pixels — one step covers a whole unit). Compare
against a random order: the learned order should climb faster.

```python
fig = plot.reveal({"RoT order": curve, "Random": random_curve})
```

```{image} _static/figures/reveal.light.png
:class: only-light
:alt: Learned order reaching full accuracy after one input versus random climbing slowly
```

```{image} _static/figures/reveal.dark.png
:class: only-dark
:alt: Learned order reaching full accuracy after one input versus random climbing slowly
```

## Rules that bite

- **Units by default.** Embedding dims (text) and channels (images)
  are revealed together. `granularity="element"` reveals individual
  numbers instead — but the value must match how the order was made.
  Tables are unaffected (one step per column).
- **Filler is trimmed.** Steps where every row has exhausted its real
  pieces are dropped; per-step accuracy only counts rows still going.
  `include_padded=True` keeps the full rectangle instead.
- **Orders carry padding.** Filler ranks last as `-1`; that is why
  `score_ordering` takes no mask. Argument order is
  `score_ordering(inputs, answers, order)` — swapping the first two
  raises a `ValueError` that says so.
- **Metrics.** Default is accuracy (any number of answers).
  `return_confusion=True` gives per-step confusion counts instead;
  `metric=` accepts a custom callable over binary counts.

The runnable version is the
[Shapes demo](notebooks/06_shapes_demo.ipynb).
