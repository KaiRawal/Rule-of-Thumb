# API reference

Every public entry point, rendered from docstrings. Start with `fit`,
then pick a modality. All of these are also listed on the [landing
page]({ref}`landing page <symbol-index>`).

## Facade

```{eval-rst}
.. autoclass:: ruleofthumb.Explainer
   :members:
```

```{eval-rst}
.. autofunction:: ruleofthumb.fit
```

(ruleofthumb.fit_tabular)=

```{eval-rst}
.. autofunction:: ruleofthumb.fit_tabular
```

```{eval-rst}
.. autofunction:: ruleofthumb.fit_text
```

```{eval-rst}
.. autofunction:: ruleofthumb.fit_image
```

```{eval-rst}
.. autofunction:: ruleofthumb.load_explainer
```

## Tuning and ingestion

```{eval-rst}
.. autofunction:: ruleofthumb.autotune
```

```{eval-rst}
.. autoclass:: ruleofthumb.AutotuneResult
   :members:
```

```{eval-rst}
.. autofunction:: ruleofthumb.embed_texts
```

```{eval-rst}
.. autoclass:: ruleofthumb.TextEmbeddings
   :members:
```

```{eval-rst}
.. autodata:: ruleofthumb.DEFAULT_TEXT_MODEL
```

```{eval-rst}
.. autodata:: ruleofthumb.DEFAULT_TEXT_REVISION
```

```{eval-rst}
.. autofunction:: ruleofthumb.load_images
```

```{eval-rst}
.. autoclass:: ruleofthumb.ImageBatch
   :members:
```

```{eval-rst}
.. autofunction:: ruleofthumb.embed_images
```

```{eval-rst}
.. autoclass:: ruleofthumb.ImageEmbeddings
   :members:
```

```{eval-rst}
.. autodata:: ruleofthumb.DEFAULT_IMAGE_MODEL
```

## Utilities

```{eval-rst}
.. autofunction:: ruleofthumb.pad_sequences
```

```{eval-rst}
.. autofunction:: ruleofthumb.lengths_to_mask
```

```{eval-rst}
.. autofunction:: ruleofthumb.pad_images
```

## Raw models

(ruleofthumb.core.RoT)=

```{eval-rst}
.. autoclass:: ruleofthumb.core.RoT
   :members:
```

```{eval-rst}
.. autoclass:: ruleofthumb.text.RoTText
   :members:
```

```{eval-rst}
.. autoclass:: ruleofthumb.image.RoTImage
   :members:
```

## Plotting

```{eval-rst}
.. automodule:: ruleofthumb.plot
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.waterfall
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.force
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.decision
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.bar
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.beeswarm
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.text_html
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.text_matplotlib
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.saliency
```

```{eval-rst}
.. autofunction:: ruleofthumb.plot.word_clouds
```
