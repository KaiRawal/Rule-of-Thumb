# Limits

Every explanation method trades something away, and an honest page
about those trades is worth more than a persuasive one. RoT's
simplicity — one shared, additive stand-in — is what makes it cheap,
query-free, and auditable from logged predictions alone. It is also,
inevitably, what blinds it to certain things. This page maps the
blind spots so you can work inside them with confidence.

## A model of averages, not arrangements

For text and images, a single shared set of weights is reused at
every position. That keeps the model tiny and position-independent,
but it means the stand-in effectively sees the *average word* of a
sentence and the *total ink per channel* of a picture:

```{image} _static/figures/limits-order.light.png
:class: only-light
:alt: Two sentences with swapped word order receiving identical explanations
```

```{image} _static/figures/limits-order.dark.png
:class: only-dark
:alt: Two sentences with swapped word order receiving identical explanations
```

- **Word order disappears.** "Dog bites man" and "man bites dog"
  receive the same explanation, because they contain the same words.
  Tasks driven by *which* words appear — sentiment, topic, resume
  screening — work beautifully; negation scope, syntax, and counting
  do not. If your question lives in the ordering, this is not the
  right tool, and it is better to learn that here than from a
  confusing result.

```{image} _static/figures/limits-ink.light.png
:class: only-light
:alt: A tight spot and a wide ring with equal total ink scoring the same
```

```{image} _static/figures/limits-ink.dark.png
:class: only-dark
:alt: A tight spot and a wide ring with equal total ink scoring the same
```

- **Position washes out on raw pixels.** A tight spot and a wide ring
  with equal total ink score the same, for exactly the same reason.
  This is why image paths travel through a backbone by default: rich
  feature maps (576 channels instead of 3) separate classes that bare
  pixels cannot. Think of raw pixels as a teaching aid for toy
  problems, not a production setting.

## Habits that keep you safe

1. **Feed it rich inputs**: transformer embeddings for text, backbone
   maps for images. The stand-in can only be as perceptive as what
   you show it.
2. **Always check agreement** (`predict` against the black box's own
   answers) before trusting a word of the explanation. Below about
   0.75, stop and reconsider — the fit will have warned you already.
3. **Narrow wide questions first.** Explaining a 1000-way classifier
   rarely means fitting all 1000 answers; remap to the question you
   actually care about (say, one class versus the rest) and fit on
   that:

```python
binary = (black_box_answers == 283).astype(np.int64)  # Persian-cat-vs-rest
exp = rot.fit_image(binary, images, mask=mask)
```

## Faithful is not the same as plausible

It helps to keep two judgements apart. RoT reproduces *the model's
answers* — that is fidelity, and the agreement score plus the reveal
curve measure it. Whether highlighted words look convincing to *a
person* is plausibility, a different question with its own baselines
(random orders, center-of-image priors, which embarrassingly often
win). A faithful ranking of what the model actually leans on can look
nothing like what you would underline yourself, especially where
averaging has discarded order or position. The
[Hatexplain](notebooks/04_hatexplain.ipynb) and
[Gaze](notebooks/05_salicon.ipynb) notebooks explore both sides
openly, baselines included.
