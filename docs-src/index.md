# ruleofthumb

> ⚠️ **Experimental 0.0.x pre-alpha.** This package was entirely vibe-coded
> from hand-written research code. It may break, and backwards-incompatible
> changes are expected before any 1.0. Verify explanations before trusting
> them.

Rule of Thumb (RoT) trains a simple, transparent surrogate on partial
observations of a black-box model's behaviour, producing per-feature
importances that can be revealed incrementally (most-important-first)
while preserving the black box's predictions.

## Install

Requires Python >= 3.9. A single install ships everything (core + plotting
+ LLM/vision helpers):

```bash
pip install ruleofthumb==0.0.1
```

From source (this repo's root):

```bash
pip install .
```

For development:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Guides

- [Quickstart](quickstart.md) — all three modalities in one page.
- [Tabular](tabular.md) · [Text](text.md) · [Image](image.md) —
  per-modality usage with worked notebooks.
- [Workflows](workflows.md) — reveal curves, tuning, persistence,
  non-linear shapes, plotting, devices.
- [Migration](migration.md) — breaking changes across the 0.x line.
- [Capacity](capacity.md) — pooled-model fidelity ceilings and backbone
  guidance. Read this before trusting an explanation.
- [Test report](test-report.md) — the maintained test-suite report.

(symbol-index)=

## Symbol index

Every public symbol, each rendered on the [API reference](api.md):

| Symbol | What it is |
|---|---|
| `fit` | Auto-detecting factory (all modalities) |
| `fit_tabular` | Tabular factory |
| `fit_text` | Text / embedding factory |
| `fit_image` | Image factory |
| `Explainer` | Fitted explainer facade |
| `load_explainer` | Reload a saved explainer |
| `autotune` | Hyperparameter search |
| `AutotuneResult` | Tuning result (`.explainer`, `.best_params`, `.trials`) |
| `embed_texts` | Transformer text embedder |
| `TextEmbeddings` | Embedding result (embeddings, mask, tokens) |
| `DEFAULT_TEXT_MODEL` | Bundled embedding model name |
| `load_images` | Image-file loader |
| `ImageBatch` | Loaded images + validity mask |
| `pad_sequences` | Ragged text → rectangular batch |
| `sentinel_mask` | Rebuild a mask from legacy `-1` padding |
| `pad_images` | Mixed-size images → one batch |
| `core.RoT` | Raw tabular model |
| `text.RoTText` | Raw text model |
| `image.RoTImage` | Raw image model |
| `plot` | Visualisations for every modality |

## Examples

Hello-world notebooks on dummy data (no downloads or GPUs needed),
re-executed on every site build under *Notebooks*:

- `examples/01_tabular_quickstart.ipynb`
- `examples/02_text_quickstart.ipynb`
- `examples/03_image_quickstart.ipynb`

## Licence

MIT — see `LICENSE`. Note the experimental caveat above still applies.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Contents

quickstart
tabular
text
image
workflows
notebooks/01_tabular_quickstart
notebooks/02_text_quickstart
notebooks/03_image_quickstart
migration
capacity
api
test-report
```
