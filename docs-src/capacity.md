# Limits

Read this before trusting an explanation. The stand-in is deliberately
simple, and simplicity has blind spots.

## It sees averages, not arrangements

For text and images, one shared set of weights is reused at every
position — so the model effectively sees the *average word* of a
sentence and the *total ink per channel* of a picture. Consequences:

```{image} _static/figures/limits-order.light.png
:class: only-light
:alt: Two sentences with swapped word order receiving identical explanations
```

```{image} _static/figures/limits-order.dark.png
:class: only-dark
:alt: Two sentences with swapped word order receiving identical explanations
```

- **Word order is invisible.** "Dog bites man" and "man bites dog"
  get the same explanation. Sentiment and topic tasks (driven by
  *which* words appear) work well; negation scope, syntax, and
  counting do not.

```{image} _static/figures/limits-ink.light.png
:class: only-light
:alt: A tight spot and a wide ring with equal total ink scoring the same
```

```{image} _static/figures/limits-ink.dark.png
:class: only-dark
:alt: A tight spot and a wide ring with equal total ink scoring the same
```

- **Position is pooled away on raw pixels.** A tight spot and a wide
  ring with equal total ink score the same. This is why image paths
  go through a backbone by default: rich feature maps (576 channels
  instead of 3) separate classes the pixels cannot.

## Practical rules

1. **Prefer rich inputs**: transformer embeddings for text, backbone
   maps for images. Raw pixels are for toy problems.
2. **Always check agreement** (`predict` vs the black box's answers)
   before trusting. Below ~0.75, stop.
3. **Many answers? Simplify first.** Explaining a 1000-way classifier
   rarely means fitting 1000 answers — remap to your question first
   (e.g. `y == target_class` for one-vs-rest), then fit on that:

```python
binary = (black_box_answers == 283).astype(np.int64)  # Persian-cat-vs-rest
exp = rot.fit_image(binary, images, mask=mask)
```

## Faithful is not plausible

RoT reproduces *the model's answers*. Whether highlighted words look
convincing to *people* is a different question with its own baselines
(random orders, center-of-image priors — which often win). A faithful
ranking of what the model uses can look nothing like what a person
would underline, especially where averaging discards order or
position. The [Hatexplain](notebooks/04_hatexplain.ipynb) and
[Gaze](notebooks/05_salicon.ipynb) notebooks show both sides honestly.
