# Plots

`ruleofthumb.plot` draws explanations for every modality. One
convention runs through all of it: **red pushes toward the predicted
answer, blue pushes away.** Every function hands you a matplotlib
`Figure` and deliberately never displays it — saving, showing, and
embedding stay your decision.

## Tables, drawn two ways

The tabular plots speak SHAP's visual language on purpose, so anyone
arriving from that world feels at home (they need the `[plot]`
extra). Waterfall, force, and decision each unpack a single answer;
bar and beeswarm summarise a batch:

```python
plot.waterfall(exp, X[:1], feature_names=names)  # one answer, stacked
plot.force(exp, X[:1], feature_names=names)      # one answer, arrows
plot.decision(exp, X[:1], feature_names=names)   # one answer, paths
plot.bar(exp, X[:50], feature_names=names)       # batch ranking
plot.beeswarm(exp, X[:50], feature_names=names)  # batch vs values
```

```{image} _static/figures/card-tabular.light.png
:class: only-light
:alt: Bar chart of per-column importances
```

```{image} _static/figures/card-tabular.dark.png
:class: only-dark
:alt: Bar chart of per-column importances
```

## Words, highlighted and gathered

Token explanations want to be read in place — the sentence itself,
with the telling words glowing — and occasionally gathered across a
whole corpus:

```python
plot.text_html(imp[0], tokens[0])       # notebook highlight
plot.text_matplotlib(imp[0], tokens[0]) # static export for papers
plot.word_clouds(imp, all_tokens)       # toward / against / combined
```

```{image} _static/figures/card-text.light.png
:class: only-light
:alt: Sentence with the important word highlighted
```

```{image} _static/figures/card-text.dark.png
:class: only-dark
:alt: Sentence with the important word highlighted
```

## Pixels, overlaid where they belong

A saliency map means most with its picture underneath it, so
`plot.saliency` layers the signed importances over the image itself
(the bare image argument is optional, for when the map alone says
enough):

```python
plot.saliency(imp[0], image=rgb)  # overlay; image optional
```

```{image} _static/figures/card-image.light.png
:class: only-light
:alt: Saliency overlay highlighting the circle
```

```{image} _static/figures/card-image.dark.png
:class: only-dark
:alt: Saliency overlay highlighting the circle
```

## Curves that prove the ranking

```python
plot.reveal({"RoT order": good, "Random": baseline})
```

Each curve is one value per reveal step — the outputs of
`score_ordering` — drawn so the learned order can be judged against
chance at a glance. Like everything on the reveal path, this
re-scores the surrogate only; your model is never queried. See
[Checking the ranking by revealing less](reveal.md).

Every call on this page appears in runnable form under
[Examples](examples.md).
