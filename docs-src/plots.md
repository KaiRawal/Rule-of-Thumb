# Plots

`ruleofthumb.plot` draws every modality. One convention everywhere:
**red pushes toward the predicted answer, blue pushes away.** Every
function returns a matplotlib `Figure` and never shows it — save or
display it yourself.

## Tables

SHAP-style plots for one answer or a batch (needs the `[plot]` extra):

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

## Words

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

## Pixels

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

## Reveal curves

```python
plot.reveal({"RoT order": good, "Random": baseline})
```

One value per reveal step per curve (the outputs of `score_ordering`).
See [Reveal curves](reveal.md).

All calls above appear runnable in [Examples](examples.md).
