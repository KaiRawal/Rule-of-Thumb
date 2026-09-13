# Examples

These notebooks actually run — every one is re-executed from scratch
on each site build, so what you read is what you would get. Begin at
the top: dummy data, plain CPU, a few seconds each.

## Gentle beginnings

| Notebook | What you will see happen |
|---|---|
| [Tabular demo](notebooks/01_tabular_quickstart.ipynb) | Tables: fitting, ranking, a reveal curve, a waterfall |
| [Text demo](notebooks/02_text_quickstart.ipynb) | Sentences: masks, highlighted words, word clouds |
| [Image demo](notebooks/03_image_quickstart.ipynb) | Pictures: raw arrays and a saliency overlay |
| [Shapes demo](notebooks/06_shapes_demo.ipynb) | Circles against empty backgrounds: saliency plus the reveal comparison that ties the story together |

## For later, when the above feels easy

Real data, real models, and honestly reported caveats — worth your
time once the mechanics feel natural:

- [HateXPlain](notebooks/04_hatexplain.ipynb): text explanations held
  up against human rationale spans — plausibility versus faithfulness
  versus bias slices.
- [Gaze](notebooks/05_salicon.ipynb): image saliency held up against
  human eye fixations, baselines included.

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
