# Examples

These notebooks actually run — every one is re-executed from scratch
on each site build, so what you read is what you would get. Begin at
the top: dummy data, plain CPU, a few seconds each.

## Gentle beginnings

| Notebook | What you will see happen |
|---|---|
| [Tabular demo](notebooks/01_tabular_quickstart.ipynb) | Tables: fitting, ranking, a reveal curve vs random, waterfall/force/decision plus batch bar/beeswarm |
| [Text demo](notebooks/02_text_quickstart.ipynb) | Sentences: masks, agreement, highlighted words, word clouds, static export, word-level reveal curve |
| [Image demo](notebooks/03_image_quickstart.ipynb) | Pictures: scores vs accuracy, magnitude view vs signed saliency overlay (plus map-only) |
| [Shapes demo](notebooks/06_shapes_demo.ipynb) | Circles against empty backgrounds: saliency plus the reveal comparison that ties the story together |

## For later, when the above feels easy

Real data, real models, and honestly reported caveats — worth your
time once the mechanics feel natural:

- [HateXPlain](notebooks/04_hatexplain.ipynb): text explanations held
  up against human rationale spans — curated highlights, per-class and
  bias-slice word clouds, AUROC histogram, deletion/insertion reveal
  curves versus random.
- [Gaze](notebooks/05_salicon.ipynb): image saliency held up against
  human eye fixations — multi-method overlays (RoT-maps/pixel, IG,
  occlusion, center), full-set metric chart, pixel-arm reveal curve.

```{toctree}
:hidden:
:maxdepth: 1

notebooks/01_tabular_quickstart
notebooks/02_text_quickstart
notebooks/03_image_quickstart
notebooks/06_shapes_demo
notebooks/04_hatexplain
notebooks/05_salicon
```
