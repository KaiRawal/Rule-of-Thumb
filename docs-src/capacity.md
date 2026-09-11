# Capacity and backbone guidance

The text and image variants share their importance weights across
positions, making each surrogate a *linear model on a pooled
representation*. This page explains the resulting fidelity ceiling and
how to work within it. (Canonical numbers: `ToDo.md` item 16;
pinned tests: `tests/integration/test_image_integration.py`.)

## Definitions

`n_classes` (`K`) is the number of possible answers (2 for cat-vs-dog, 10
for digits 0-9). `embedding` (`E`) is the length of the number-list
describing one token (e.g. 384). `channels` (`C`) is the number of layers
per pixel (1 for grey, 3 for colour, 576 for MobileNetV3-Small feature
maps). `tokens` (`T`) is the number of word-pieces per text;
`height`/`width` (`H`/`W`) are image dimensions.

## The pooled model

The learned slopes `a` and offsets `b` have shape `(n_classes,
embedding)` for text and `(n_classes, channels)` for images, and the same
`a`/`b` row is reused at every token/pixel (`2·K·C` parameters,
independent of spatial size). Predictions therefore depend on the
**token-mean embedding** (`E` averaged numbers) for text and on
**per-channel spatial sums** (`C` totals) for images.

Sharing is what makes saliency maps position-invariant (where a pattern
appears matters less than how much of it there is) and keeps the learned
count independent of `T`/`H`/`W`. Fidelity is then bounded by how well
classes separate in that pooled space.

## Measured ceilings (digits, pinned by tests)

- Raw single-channel images (`C = 1`) pool to total ink mass alone:
  10-class fidelity floors near the majority baseline (always guessing the
  commonest class) no matter how long the fit runs
  (`test_multiclass_confusion_counts_and_reveal_curve`).
- Coordinate channels (`{mass, row-mass, col-mass}`) lift digits to ~0.35
  (`test_coordinate_channels_restore_multiclass_capacity`); the TinyCNN
  trunk itself (`(500, 8, 8, 8)` maps) reaches only ~0.49 and is not used
  as a backbone.
- The same shared-weight image surrogate reaches ~0.99 on 10-class digits
  from live MobileNetV3-Small `(500, 576, 7, 7)` maps with 500 images
  exceeding 49 spatial positions
  (`test_multiclass_rich_backbone_strong_accuracy`, strong bar `>= 0.8`).

For text the analogue holds: pooling to the token-mean makes word order
invisible, so tasks driven by syntax, negation scope or token counts hit a
ceiling even with high-dimensional embeddings, while lexical tasks (e.g.
sentiment) and rich pretrained embeddings stay strong.

## Practical rules

1. **Prefer rich channel representations**: pretrained transformer
   embeddings for text, MobileNet (or similar) feature maps for images.
2. **Always check surrogate predicted-class accuracy** (`predict` vs the
   black box's labels) before trusting an explanation.
3. Rectangular batches only: use `pad_sequences` / `pad_images` plus masks
   for ragged or mixed-size data.
4. The deferred opt-in per-location variant (unshared weights) is tracked
   in `ToDo.md` item 16 — only reach for it if raw-input multiclass is
   required.

## Many-class heads

Explaining a pretrained classifier with hundreds of classes (e.g. a
1000-way ImageNet head) rarely means fitting a 1000-way surrogate: that is
`2·K·C` parameters, and out-of-distribution inputs tend to collapse onto a
single predicted class anyway. Remap the black-box labels first — either
binary-vs-rest (`y == target_class`) or a top-`k` subset — then fit (and
autotune) on the remapped labels; `n_classes` infers itself from them:

```python
binary = (black_box_labels == 283).astype(np.int64)  # Persian-cat-vs-rest
exp = rot.fit_image(y_outputs=binary, x_inputs=images, mask=mask)
result = rot.autotune(y_outputs=binary, x_inputs=images, mask=mask, seed=0)
```

Passing labels outside `[0, n_classes)` fails fast with a `ValueError`
naming the offending class, instead of a torch indexing error.

## Fidelity vs plausibility

RoT reproduces *black-box outputs*: the number to check first is always
surrogate-vs-black-box agreement on your inputs. Agreement with *human*
rationales (which tokens or regions people find meaningful) is a separate
evaluation with its own baselines — always include trivial controls such
as position or center priors, which regularly beat learned maps on
human-agreement scores, and always compare deletion/insertion curves
against random-mask baselines. A faithful ranking of what the model uses
can still look nothing like what a person would highlight, especially
where pooling discards order or location (see above) or the black box
itself is near-degenerate. Tracked as `ToDo.md` item 24.
