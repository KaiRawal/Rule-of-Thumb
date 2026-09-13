# Checking the ranking by revealing less

A ranking is only as good as its proof, and the reveal curve is that
proof — strictly optional, run only when you ask for it. Uncover each
row's inputs most-important-first and re-check the answer at every
step, and see how little it takes to hold steady. A ranking you can
believe in keeps the *stand-in's* answer after revealing very little;
a random one needs most of the input. Nothing here queries your
model: every step re-scores the surrogate, so this tests the ranking,
never the black box. Three calls tell the whole story:

```python
order = exp.get_order(inputs)                          # best first
preds = exp.ordered_predict(inputs, order)             # answers per step
curve = exp.score_ordering(inputs, answers, order)     # accuracy per step
```

`curve[k]` is the accuracy after uncovering the top-`k` pieces —
columns, words, or pixels, where one step always covers a whole unit.
Drawn against a random order, the learned curve should climb earlier
and stay there:

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

## Details worth knowing before you rely on this

- **Units travel together by default.** A word's embedding dimensions
  (text) and a pixel's channels (images) are revealed as one.
  `granularity="element"` reveals individual numbers instead — but
  the value must match how the order was produced, since nothing
  detects a mismatch for you. Tables are unaffected: one step per
  column, always.
- **Filler is trimmed, not plotted.** Steps where every row has
  exhausted its genuine pieces are dropped, and per-step accuracy
  counts only the rows still going. `include_padded=True` keeps the
  full rectangle instead, if you ever want it.
- **Orders already carry their padding.** Filler ranks last as `-1`,
  which is why `score_ordering` takes no mask. Mind the argument
  order — `score_ordering(inputs, answers, order)` — since swapping
  the first two raises a `ValueError` that tells you exactly that.
- **Accuracy is the default lens.** It works for any number of
  answers. `return_confusion=True` gives per-step confusion counts
  instead, and `metric=` accepts a custom callable over binary
  counts for the two-answer case.

To watch this run on real (if tiny) data, the
[Shapes demo](notebooks/06_shapes_demo.ipynb) ends with exactly this
comparison.
