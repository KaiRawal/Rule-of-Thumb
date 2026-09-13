# Core ideas

Everything else in these docs builds on seven terms. Each takes a
paragraph; none assumes you have met explainability before.

**Black box.** Anything that maps inputs to answers: `y = model(X)`.
A scikit-learn classifier, a sentiment head on a language model, an
image network behind an API. RoT never opens it — it only watches
what it answers, which is why a model you cannot inspect is still
perfectly explainable.

**Stand-in (surrogate).** A deliberately simple model, fitted to copy
the black box's answers on your data. You create one per modality —
`fit_tabular`, `fit_text`, or `fit_image` (or `fit`, which picks from
the shape of what you hand it) — and the returned `Explainer` wraps
it up with everything you need to interrogate it.

**Importance.** One signed number per input piece: a column, a word,
a pixel. Positive means evidence *toward* the predicted answer,
negative means evidence *against* — the red-against-blue convention
you will see in every plot. For two-answer tasks you get the numbers
for the second answer directly; where there are more answers, you get
one set per answer.

**Order.** `get_order` sorts each row's pieces most-important-first,
by absolute importance. It answers the most practical question in the
whole package: "what should I look at first?"

**Reveal curve.** `ordered_predict` uncovers pieces in that order and
re-scores the answer at each step, while `score_ordering` summarises
how faithful the ranking is — usually accuracy per step. A ranking
you can believe in keeps the right answer after revealing very
little; the [Shapes demo](notebooks/06_shapes_demo.ipynb) shows this
happening live.

**Mask.** Batches are rectangles, but sentences and pictures come in
all shapes — so short rows get meaningless filler alongside a boolean
mask saying what is real (`True` means a genuine word or pixel). The
filler value itself never matters; the mask carries the truth. Padded
slots always score exactly zero and rank last, where they show up as
`-1` in orders.

```{image} _static/figures/mask.light.png
:class: only-light
:alt: Three sentences padded into a rectangle with a green mask marking real tokens
```

```{image} _static/figures/mask.dark.png
:class: only-dark
:alt: Three sentences padded into a rectangle with a green mask marking real tokens
```

**Agreement.** `exp.train_agreement_` is the stand-in's accuracy
against the black box's answers on the training inputs. Treat it as a
tripwire rather than a verdict: it is measured in-sample, so it
flatters a little. Near 1, carry on. Below about 0.75 the fit warns
you outright, and the numbers may be gibberish.
[Limits](capacity.md) walks through the usual causes.

## Why predictiveness instead of perturbation?

Most explanation methods work by poking the model: change an input a
little, add up how much the output moves, and call the twitchiest
features important. That is a fine strategy when you hold the model's
weights in your hand and can query it endlessly for free. But think
about the situations that motivated this package. A commercial API
charges per call and hides its weights entirely; asking it to score
hundreds of doctored inputs per explanation — the going rate is
roughly 500 for SHAP and 5000 for LIME — is slow at best and ruinous
at worst. And when a model answers a flat yes or no, small nudges
often change nothing at all, leaving a perturbation method with
nothing to measure.

RoT takes the other fork in the road. Instead of asking "how would
the answer change if I altered this input", it asks "how should what
I already know change my prediction of the system's behaviour" — and
it learns that mapping once, for the whole dataset, from the answers
you have already collected. No weights required, no extra queries,
no refit per row. One fitted explainer then answers every question:
importances, rankings, reveal curves, plots. If you want the full
argument, with experiments across language models, proprietary-system
audits, and scientific discovery, the
[paper](https://arxiv.org/abs/2608.10766) is the place to go; the
rest of these docs is the hands-on version.
