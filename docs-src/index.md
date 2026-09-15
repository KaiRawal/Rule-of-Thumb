# Rule of Thumb

[![PyPI](https://img.shields.io/pypi/v/ruleofthumb-rot)](https://pypi.org/project/ruleofthumb-rot/)
[![Docs](https://img.shields.io/readthedocs/rule-of-thumb)](https://rule-of-thumb.readthedocs.io/)
[![License](https://img.shields.io/github/license/KaiRawal/Rule-of-Thumb)](https://github.com/KaiRawal/Rule-of-Thumb/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/ruleofthumb-rot)](https://pypi.org/project/ruleofthumb-rot/)

Suppose you have a model you cannot open. It takes inputs, returns
answers, and never says why. Perhaps it is a language model behind an
API, a vendor's scoring system, or simply a pipeline nobody fully
understands anymore. You would like to trust a particular answer —
but all you can see is the answer itself.

Rule of Thumb (RoT) is built for exactly that situation. It learns a
small, transparent stand-in that imitates the model's answers on your
data, then tells you, for each answer, which inputs mattered and in
which direction. If you know scikit-learn, think
`feature_importances_` — except the numbers are **signed** (positive
pushes *toward* the predicted answer, negative pushes *away*), and
they arrive with a ranking, so you can uncover inputs
most-important-first and watch the answer hold steady.

```{image} _static/figures/hero.light.png
:class: only-light
:alt: Four-step diagram: unopenable model, simple stand-in, what mattered, same answer with fewer inputs
```

```{image} _static/figures/hero.dark.png
:class: only-dark
:alt: Four-step diagram: unopenable model, simple stand-in, what mattered, same answer with fewer inputs
```

- 📄 Paper: https://arxiv.org/abs/2608.10766
- 🌐 Project website: https://kairawal.github.io/Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information/website/
- 🔬 Research code: https://github.com/KaiRawal/Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information

## Why this approach exists

Modern models are often easiest to use exactly where they are hardest
to inspect. A language model can classify resumes or reviews with no
training data at all — yet deciding whether to *deploy* it still means
labelling data by hand and checking agreement, which quietly erases
the very convenience that made the model attractive. Explanations were
supposed to offer a second opinion: show me *which* inputs the answer
rested on, and I can judge whether the reasoning looks sound.

The trouble is that classic explanation methods assume access you may
not have. Some need the model's weights and gradients, which a
closed API will never share. Others need to interrogate the model
hundreds or thousands of times with doctored inputs — SHAP asks for
roughly 500 extra queries per explanation, LIME more like 5000 —
which is slow at best and, behind a paywalled API, ruinous. And when a
model answers a plain yes or no, tiny perturbations often change
nothing at all, leaving sensitivity-based methods with nothing to
measure.

RoT sidesteps all of this by asking a different question. Rather than
"how would the answer change if I altered this input", it asks "how
should what I already know change my prediction of the system's
behaviour" — learning, once for the whole dataset, which feature
values are genuinely predictive of the answers you have already
collected. No weights required, no extra queries, no per-row refit.
That is what makes it practical for zero-shot language-model
decisions, for auditing proprietary systems you can only observe, and
for scientific settings where you need explanations you can defend.
The [paper](https://arxiv.org/abs/2608.10766) works through each of
these in depth; what follows here is the hands-on version.

## Three modalities, one pattern

```{image} _static/figures/card-tabular.light.png
:class: only-light
:alt: Bar chart of per-column importances, red toward the answer and blue against
```

```{image} _static/figures/card-tabular.dark.png
:class: only-dark
:alt: Bar chart of per-column importances, red toward the answer and blue against
```

**Tabular** — rows and columns, like any sklearn dataset, when you
want to know which columns carried each decision. One number per
column:

```python
exp = rot.fit_tabular(y_answers, X_table)
imp = exp.get_explanation(X_table)  # [N, columns]
```

```{image} _static/figures/card-text.light.png
:class: only-light
:alt: Sentence with the word wonderful highlighted red
```

```{image} _static/figures/card-text.dark.png
:class: only-dark
:alt: Sentence with the word wonderful highlighted red
```

**Text** — sentences in, one number per word out. This is the natural
home for those closed language-model APIs: hand over raw strings and
learn which words the answers rested on, padding handled for you:

```python
exp = rot.fit_text(y_answers, ["a wonderful film", "terrible pacing"])
imp = exp.get_explanation(["a wonderful film", "terrible pacing"])  # [N, words]
```

```{image} _static/figures/card-image.light.png
:class: only-light
:alt: Circle image with red saliency overlay on the circle
```

```{image} _static/figures/card-image.dark.png
:class: only-dark
:alt: Circle image with red saliency overlay on the circle
```

**Images** — pictures in, one number per pixel out, for classifiers
you would rather not query hundreds of extra times. Pass file paths;
they are embedded through a frozen backbone:

```python
exp = rot.fit_image(y_answers, ["cat.jpg", "dog.jpg"])
imp = exp.get_explanation(["cat.jpg", "dog.jpg"])  # [N, H, W]
```

## Install

```bash
pip install ruleofthumb-rot
pip install "ruleofthumb-rot[text]"    # raw-sentence support
pip install "ruleofthumb-rot[image]"   # image files and backbones
pip install "ruleofthumb-rot[plot]"    # visualisations
```

Tracking unreleased `main`? See [Install](install.md) for the source
install and version notes.

:::{warning}
The numbers imitate the *model*, not the world. Every fit reports how
faithfully the stand-in reproduces the model's answers — please check
that score before believing an explanation. See [Limits](capacity.md).
:::

## Next steps

- If you are new here, [Get started](start.md) walks you through
  installing, fitting your first explanation in about five minutes,
  and meeting the core ideas.
- When you are ready to use it properly, the [Guide](guide.md) covers
  each modality in depth, plus reveal curves, plots, tuning, and
  saving your work.
- Prefer runnable code? [Examples](examples.md) all run on dummy data
  in seconds, and the [API reference](api.md) documents every public
  function.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Get started

start
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Guide

guide
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Examples

examples
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

api
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project

releases
migration
development
```
