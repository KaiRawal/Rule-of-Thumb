# Rule of Thumb

[![PyPI](https://img.shields.io/pypi/v/ruleofthumb-rot)](https://pypi.org/project/ruleofthumb-rot/)
[![Docs](https://img.shields.io/readthedocs/rule-of-thumb)](https://rule-of-thumb.readthedocs.io/)
[![License](https://img.shields.io/github/license/KaiRawal/Rule-of-Thumb)](https://github.com/KaiRawal/Rule-of-Thumb/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/ruleofthumb-rot)](https://pypi.org/project/ruleofthumb-rot/)

You have a model you can't open — it takes inputs and returns answers,
and you want to know *why* it answered that way. Rule of Thumb (RoT)
learns a simple stand-in that copies the model's answers, then tells you,
for each answer, which inputs mattered and in which direction.

If you know scikit-learn, think `feature_importances_` — except the
numbers are **signed** (positive pushes *toward* the predicted answer,
negative pushes *away*) and they come with a ranking that lets you
reveal inputs most-important-first while keeping the same answer.

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

## Three modalities, one pattern

```{image} _static/figures/card-tabular.light.png
:class: only-light
:alt: Bar chart of per-column importances, red toward the answer and blue against
```

```{image} _static/figures/card-tabular.dark.png
:class: only-dark
:alt: Bar chart of per-column importances, red toward the answer and blue against
```

**Tabular** — rows and columns, like any sklearn dataset.
One number per column:

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

**Text** — sentences in, one number per word out.
Pass raw strings; padding is handled for you:

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

**Images** — pictures in, one number per pixel out.
Pass file paths; they are embedded through a frozen backbone:

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
The numbers copy the *model*, not the truth. Every fit reports an
agreement score against the model's answers — check it before believing
an explanation. See [Limits](capacity.md).
:::

## Next steps

- New here? Start at [Get started](start.md): install, a 5-minute
  first explanation, and the core ideas.
- Ready to use it? The [Guide](guide.md) covers each modality plus
  reveal curves, plots, tuning, and saving.
- Want runnable code? [Examples](examples.md) runs on dummy data in
  seconds; [API reference](api.md) documents every public function.

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
