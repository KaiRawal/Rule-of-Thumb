# Plots

`ruleofthumb.plot` draws explanations for every modality. One
convention runs through all of it: **red pushes toward the predicted
answer, blue pushes away.** Every function hands you a matplotlib
`Figure` and deliberately never displays it — saving, showing, and
embedding stay your decision:

```python
fig = plot.waterfall(exp, X[:1], feature_names=names)
fig.savefig("waterfall.png")  # or fig.show(), display(fig), ...
```

All snippets below are copy-paste runnable (CPU, seconds) once you
have a fitted explainer. Each names the notebook cell where it runs
end to end under [Examples](examples.md) — those notebooks are
re-executed on every docs build, so what you read is what you get.
The static cards alongside are pre-generated illustrations from
`docs-src/_figures/generate_figures.py` (synthetic dummy data, not
fitted explanations).

## Tables, drawn two ways

The tabular plots speak SHAP's visual language on purpose, so anyone
arriving from that world feels at home (they need the `[plot]`
extra: `pip install ruleofthumb-rot[plot]`). Start from a fitted
table explainer:

```python
import numpy as np
import ruleofthumb as rot
from ruleofthumb import plot

rng = np.random.RandomState(0)
X = rng.rand(1000, 4).astype(np.float32)
y_answers = ((X[:, 0] + X[:, 1]) > 1.0).astype(np.int64)
exp = rot.fit_tabular(y_answers, X, epochs=30, seed=0)
names = ["size", "age", "zip", "color"]
```

Waterfall, force, and decision each unpack a **single answer**;
bar and beeswarm summarise a **batch**:

```python
plot.waterfall(exp, X[:1], feature_names=names)  # one answer, stacked contributions
plot.force(exp, X[:1], feature_names=names)      # one answer, arrows
plot.decision(exp, X[:1], feature_names=names)   # one answer, paths
plot.bar(exp, X[:50], feature_names=names)       # batch ranking (mean |importance|)
plot.beeswarm(exp, X[:50], feature_names=names)  # batch importances vs feature values
```

Options you will reach for: `max_display=` (cap the rows drawn),
`class_idx=` (which answer to explain when `n_classes > 2`).
Baseline semantics: the SHAP base value maps to the RoT class bias
`g_k`, and the plotted values are exactly `get_explanation()` —
RoT explains its own surrogate of the black box, not the box
directly.

Runnable in full: [Tabular demo](notebooks/01_tabular_quickstart.ipynb)
(`waterfall`, `bar`, `beeswarm`, plus `force`/`decision` and a
`reveal` comparison — see below).

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
whole corpus. Text scores are length-normalised means over tokens,
so token importances do **not** sum to the score; these views show
raw signed token weights only (no waterfall/force for text):

```python
texts = ["a wonderful film", "terrible pacing"]
out = rot.embed_texts(texts)  # embeddings, mask, tokens
exp = rot.fit_text(y_answers, out.embeddings, mask=out.attention_mask)
imp = exp.get_explanation(out.embeddings, mask=out.attention_mask)  # [N, T]

plot.text_html(imp[0], out.tokens[0])       # notebook highlight (IPython HTML or raw str)
plot.text_matplotlib(imp[0], out.tokens[0]) # static export for papers
plot.word_clouds(imp, out.tokens)           # toward (red) / against (blue) / combined
```

Useful knobs: `max_tokens=` keeps the top-|weight| tokens in order
for long sentences; `word_clouds(..., stopwords=[...], seed=0)`
filters glue words and fixes the layout. One-sided importances
render fine — the empty panel shows a "no tokens" label.

Runnable in full: [Text demo](notebooks/02_text_quickstart.ipynb)
(all three on dummy data) and
[HateXPlain](notebooks/04_hatexplain.ipynb) (curated highlights,
per-class clouds, AUROC histogram, deletion/insertion reveal
curves on real data).

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
enough). One importance number per pixel, channels already summed:

```python
paths = ["cat.jpg", "dog.jpg"]
exp = rot.fit_image(y_answers, paths)
imp = exp.get_explanation(paths)  # [N, H, W], signed, on the map grid

plot.saliency(imp[0], image=rgb)  # overlay; image optional
plot.saliency(imp[0])             # map alone
```

Tuning the look: `power=` (sign-preserving exponent on magnitudes),
`trim=` (percentile clip per side), `size=` for new figures, `ax=`
to draw into existing axes. Saliency overlays render at full
strength even when the surrogate barely tracks the box — check
`exp.train_agreement_` (warns below 0.75) and [Limits](capacity.md)
before trusting a map; it is an explanation, not a detector.

Runnable in full: [Shapes demo](notebooks/06_shapes_demo.ipynb)
(circles, no downloads) and
[Image demo](notebooks/03_image_quickstart.ipynb) (raw arrays,
magnitude view vs signed overlay), with the advanced
[Gaze](notebooks/05_salicon.ipynb) comparing RoT maps against
pixel-arm, IG, occlusion, center-prior and random baselines.

```{image} _static/figures/card-image.light.png
:class: only-light
:alt: Saliency overlay highlighting the circle
```

```{image} _static/figures/card-image.dark.png
:class: only-dark
:alt: Saliency overlay highlighting the circle
```

## Curves that prove the ranking

Each curve is one value per reveal step — the outputs of
`score_ordering` — drawn so the learned order can be judged against
chance at a glance. Like everything on the reveal path, this
re-scores the surrogate only; your model is never queried. See
[Checking the ranking by revealing less](reveal.md):

```python
import torch

points = torch.from_numpy(X[:200])
labels = torch.from_numpy(y_answers[:200])
order = exp.get_order(points)                          # best first
curve = exp.score_ordering(points, labels, order)      # accuracy per step
random_order = torch.argsort(torch.rand_like(points.float()), dim=1)
random_curve = exp.score_ordering(points, labels, random_order)

fig = plot.reveal({"RoT order": curve, "Random": random_curve})
fig = plot.reveal([curve, random_curve], labels=["RoT order", "Random"])
```

`plot.reveal` accepts a single curve, a list plus `labels=`, or a
`{name: curve}` mapping; torch tensors are accepted. One step
always covers a whole unit (column / word / pixel).

Runnable in full: [Shapes demo](notebooks/06_shapes_demo.ipynb)
(pixels), [Tabular demo](notebooks/01_tabular_quickstart.ipynb)
(columns, binary + multiclass), [Text demo](notebooks/02_text_quickstart.ipynb)
(words), [HateXPlain](notebooks/04_hatexplain.ipynb)
(deletion/insertion vs random) and [Gaze](notebooks/05_salicon.ipynb)
(RoT vs center/random).

```{image} _static/figures/reveal.light.png
:class: only-light
:alt: Learned order reaching full accuracy after one input versus random climbing slowly
```

```{image} _static/figures/reveal.dark.png
:class: only-dark
:alt: Learned order reaching full accuracy after one input versus random climbing slowly
```

## Where each visualisation runs

| Function | Minimal inputs | Full runnable demo |
|---|---|---|
| `waterfall` | `exp`, `X[:1]` | [Tabular demo](notebooks/01_tabular_quickstart.ipynb), [Quickstart](quickstart.md) |
| `force` | `exp`, `X[:1]` | [Tabular demo](notebooks/01_tabular_quickstart.ipynb) |
| `decision` | `exp`, `X[:1]` | [Tabular demo](notebooks/01_tabular_quickstart.ipynb) |
| `bar` | `exp`, `X[:50]` | [Tabular demo](notebooks/01_tabular_quickstart.ipynb) |
| `beeswarm` | `exp`, `X[:50]` | [Tabular demo](notebooks/01_tabular_quickstart.ipynb) |
| `text_html` | `imp[0]`, `tokens[0]` | [Text demo](notebooks/02_text_quickstart.ipynb), [HateXPlain](notebooks/04_hatexplain.ipynb) |
| `text_matplotlib` | `imp[0]`, `tokens[0]` | [Text demo](notebooks/02_text_quickstart.ipynb), [HateXPlain](notebooks/04_hatexplain.ipynb) |
| `word_clouds` | `imp`, `all_tokens` | [Text demo](notebooks/02_text_quickstart.ipynb), [HateXPlain](notebooks/04_hatexplain.ipynb) |
| `saliency` | `imp[0]`, `image?` | [Shapes demo](notebooks/06_shapes_demo.ipynb), [Image demo](notebooks/03_image_quickstart.ipynb), [Gaze](notebooks/05_salicon.ipynb) |
| `reveal` | `score_ordering` curves | [Shapes demo](notebooks/06_shapes_demo.ipynb), [Tabular demo](notebooks/01_tabular_quickstart.ipynb), [Text demo](notebooks/02_text_quickstart.ipynb), [HateXPlain](notebooks/04_hatexplain.ipynb), [Gaze](notebooks/05_salicon.ipynb) |
