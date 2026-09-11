# Rule of Thumb

[![PyPI](https://img.shields.io/pypi/v/ruleofthumb-rot)](https://pypi.org/project/ruleofthumb-rot/)
[![Docs](https://img.shields.io/readthedocs/rule-of-thumb)](https://rule-of-thumb.readthedocs.io/)
[![License](https://img.shields.io/github/license/KaiRawal/Rule-of-Thumb)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/ruleofthumb-rot)](https://pypi.org/project/ruleofthumb-rot/)

> ⚠️ **Experimental 0.0.x pre-alpha.** This package was entirely vibe-coded
> from hand-written research code. It may break, and backwards-incompatible
> changes are expected before any 1.0. Verify explanations before trusting
> them.

Explaining AI systems using partial information — a pip-installable
library consolidating the Rule of Thumb (RoT) explainer into one package.

- 📄 Paper: https://arxiv.org/abs/2608.10766
- 🌐 Project website: https://kairawal.github.io/Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information/website/
- 🔬 Research code: https://github.com/KaiRawal/Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information
- 📚 Documentation: https://rule-of-thumb.readthedocs.io/

RoT trains a simple, transparent surrogate ("rule of thumb") on partial
observations of a black-box model's behaviour. The surrogate attributes the
model's output to input features, producing feature importances that can be
revealed incrementally (most-important-first) while preserving the black box's
predictions.

This package consolidates the three research variants of RoT into one
installable library:

- **Tabular** (`rot.fit_tabular`): vector inputs, one importance
  weight per feature.
- **Text / LLM embeddings** (`rot.fit_text`): token-by-embedding
  inputs with padding masks and length-normalised scores.
- **Images** (`rot.fit_image`): importance shared across spatial
  locations of CNN feature maps.

`rot.fit` auto-detects the modality from the input shape; all
factories return a fitted `Explainer`.

## Install

Requires Python >= 3.9. A single install ships everything (core + plotting +
LLM/vision helpers):

```bash
pip install ruleofthumb-rot
```

Code imports the package as `ruleofthumb` (conventionally `import
ruleofthumb as rot`); only the distribution name carries the `-rot`
suffix.

Or from this repository:

```bash
git clone https://github.com/KaiRawal/Rule-of-Thumb
cd Rule-of-Thumb
pip install .
```

For local development:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Full contributor guide (checks, CI, releasing):
`docs-src/development.md`, rendered at
https://rule-of-thumb.readthedocs.io/en/latest/development.html.

To build the docs site locally (from the repo root, one command — stages
the `examples/` notebooks under `docs-src/notebooks/` (gitignored),
re-executes them, and validates links; output goes to a temp dir, nothing
is committed):

```bash
pip install -e ".[docs]" && mkdir -p docs-src/notebooks && cp examples/0*.ipynb docs-src/notebooks/ && sphinx-build -W docs-src /tmp/rot-site
```

## Usage

### Tabular data

Wrap any black-box model's outputs on your training inputs; the explainer fits
the surrogate and returns per-feature importances.

```python
import numpy as np
import ruleofthumb as rot

X_train = np.random.rand(1000, 4).astype(np.float32)
black_box_probs = (X_train[:, 0] > 0.5).astype(np.int64)  # e.g. model.predict(X_train)

exp = rot.fit(y_outputs=black_box_probs, x_inputs=X_train)   # or fit_tabular(...)
importances = exp.get_explanation(X_train)  # signed, shape [N, d]; positive = evidence toward class 1
```

### Text / token embeddings

Inputs are `(N, tokens, embedding)` float arrays. Padding is explicit: pass a
boolean validity `mask` (`True` = real token). HuggingFace `attention_mask`
tensors compose directly as `mask=`.

```python
import numpy as np
import ruleofthumb as rot
from ruleofthumb.text import lengths_to_mask, pad_sequences

# Ragged inputs? Pad them — any fill value works, the mask carries the truth:
sequences = [np.random.rand(t, 384).astype(np.float32) for t in (20, 14, 17, 9)]
x, lengths = pad_sequences(sequences)                # x: (4, 20, 384)
labels = np.array([1, 0, 1, 0], dtype=np.int64)      # e.g. LLM predictions per text
mask = lengths_to_mask(lengths, x.shape[1]).numpy()  # (4, 20) boolean validity mask

exp = rot.fit_text(y_outputs=labels, x_inputs=x.numpy(), mask=mask)
token_importances = exp.get_explanation(x.numpy(), mask=mask)
# signed, shape [N, max_tokens]: positive = evidence toward class 1; padded tokens score exactly 0
# (for n_classes > 2 the output is per-class instead: [N, n_classes, max_tokens])
```

Already have a rectangular batch and your own mask? Pass it directly as
`mask=` to `fit_text` / `get_explanation` / `get_order`. `score_ordering`
takes no mask: padding is already encoded as `-1` entries in the order.

Starting from raw strings? Pass them straight in — `fit_text` embeds them
(bundled default: `answerdotai/ModernBERT-base`, overridable via
`tokenizer=` / `model=`), derives padding automatically, and every explainer
method accepts the same strings back:

```python
import ruleofthumb as rot

exp = rot.fit_text(y_outputs=labels, x_inputs=["a wonderful film", "terrible pacing"])
token_importances = exp.get_explanation(["a wonderful film", "terrible pacing"])
order = exp.get_order(["a wonderful film", "terrible pacing"])   # reveal pipeline works on strings too
```

Need the intermediate arrays (e.g. decoded tokens for plotting)? Use
`rot.embed_texts` directly:

```python
out = rot.embed_texts(["a wonderful film", "terrible pacing"])
exp = rot.fit_text(y_outputs=labels, x_inputs=out.embeddings,
                           mask=out.attention_mask)
out.tokens  # decoded token strings, aligned with per-token importances
```

### Images

Inputs are `(N, channels, height, width)` tensors; importance is shared across
spatial locations. Mixed-size batches are supported two ways:

```python
import numpy as np
import torch
import ruleofthumb as rot
from ruleofthumb.image import pad_images

images = [np.random.rand(3, h, w).astype(np.float32) for h, w in [(32, 32), (28, 40)]]
labels = torch.randint(0, 2, (2,))

# Option A: pad into one batch, pass the validity mask, use the explainer
x, mask = pad_images(images)                          # x: (2, 3, 32, 40); mask: (2, 32, 40)
exp = rot.fit_image(y_outputs=labels, x_inputs=x.numpy(), mask=mask.numpy())
imp = exp.get_explanation(x.numpy(), mask=mask.numpy())  # signed, shape [N, H, W]; padded pixels score exactly 0

# Option B: loop over unpadded samples one at a time with the raw model
# (weights are size-agnostic, so no padding or mask is needed per sample)
from ruleofthumb.image import RoTImage
model = RoTImage(classes=2, sample_shape=(3,))
model.fit(torch.from_numpy(x), labels, epochs=50, batch_size=2, lr=0.01, mask=mask)
for img in images:
    single_imp = model.importance(torch.from_numpy(img[None]))
# raw-model importances are per-element (N, K, C, H, W); get_explanation
# reduces over classes/channels to per-pixel saliency (N, H, W)
```

The image `get_order` preserves the spatial layout `(N, H, W)` (flat pixel
indices, `-1` for padded pixels) — unlike tabular `(N, D)` and text `(N, T)`.
Starting from image files? Pass the paths straight in — `fit_image` embeds
them through a frozen backbone by default (`mobilenet_v3_small`; weights
download once), derives validity masks automatically, and every explainer
method accepts the same paths back:

```python
import ruleofthumb as rot

paths = ["cat.jpg", "dog.jpg"]
exp = rot.fit_image(y_outputs=labels, x_inputs=paths)                        # 576-channel maps
exp = rot.fit_image(y_outputs=labels, x_inputs=paths, size=(64, 64))         # resize + centre-crop first
exp = rot.fit_image(y_outputs=labels, x_inputs=paths, backbone=None)         # raw RGB pixels instead
imp = exp.get_explanation(paths)   # signed, shape [N, h, w] on the map grid
```

Prefer maps: raw pixels pool to per-channel ink mass, which caps fidelity
on focal tasks (0.61 on real pathology vs 0.95 on backbone maps). A torch
module supplies a custom trunk; `transform=` cannot be combined with a
backbone. The backbone id is recorded in `explainer.backbone` and the save
file; reloaded explainers consume map arrays.

Need custom preprocessing (e.g. ImageNet normalisation for a torchvision
black box)? Supply `transform=` (a PIL Image → tensor callable) with
`backbone=None`, or use `rot.load_images(paths, ...)` directly to inspect
`.images` / `.mask`.

### Automatic hyperparameter tuning

`rot.autotune` searches the training hyperparameters
(`learning_rate`, `batch_size`, `epochs`, `dropout_rate`, `weight_decay`)
with a seeded validation split, scores candidates by held-out reveal
fidelity, and returns the winner refit on all data:

```python
result = rot.autotune(y_outputs=labels, x_inputs=x, search="random",
                              n_candidates=8, seed=0)
result.explainer    # best config refit on all data — use like any explainer
result.best_params  # winning hyperparameters
result.trials       # every candidate with its validation score, best-first
```

`search="grid"` enumerates a `space=` dict exhaustively; `space=` accepts any
subset of the defaults. Works for all three modalities, including raw strings
and image paths. `n_classes` is inferred from the labels; any other factory
keyword (`nonlinear`, `l1_penalty`, `mask`, ...) is forwarded to both the
candidate fits and the final refit.

The factory defaults (`epochs=500, batch_size=5000`) suit tiny inputs.
Scale them to the data: small tabular batches fit anywhere, while 96px
images want `batch_size=16–64` with modest epochs. A `batch_size` larger
than N warns (training runs full-batch).

### Saving and loading

Fitted explainers round-trip through `Explainer.save` / `load_explainer`
(weights + configuration only — no refitting, no pickled classes):

```python
exp.save("explainer.rotx")
loaded = rot.load_explainer("explainer.rotx", device="cpu")
np.allclose(exp.get_explanation(x), loaded.get_explanation(x))  # identical
```

Native string / file-path ingestion is not persisted: a reloaded explainer
consumes numeric arrays (refit from strings/paths to restore it).

### Non-linear additive explanations

By default every RoT surrogate is linear in its inputs. Pass `nonlinear=` to
any factory (all three modalities, binary and multiclass) to learn a shared
elementwise response function `s` inside the additive model
`imp[k,i] = a[k,i]·(s(x[i]) + b[k,i])` — per-feature non-linear shape
curves with a parameter budget independent of input size:

```python
exp = rot.fit_tabular(y, x, nonlinear="rbf")            # Gaussian bumps
exp = rot.fit_text(y, texts, nonlinear="hinge")         # SELU hinges
exp = rot.fit_image(y, paths, nonlinear={"type": "rbf", "n_bases": 32})
```

Both responses are residual and zero-initialised, so an unfitted non-linear
model is exactly the linear one and training adds non-linearity only where
it reduces loss. Explanations stay exactly additive per feature element, so
plotting, reveal curves and persistence work unchanged; the chosen
configuration is stored in save files. Omitting `nonlinear` keeps the plain
linear model.

### Plotting

`rot.plot` renders every modality. Colour convention throughout:
**red = evidence toward the explained class, blue = against**.

```python
import ruleofthumb.plot as plot

# Tabular — SHAP's signature plots, rendered via the shap package:
plot.waterfall(exp, x[:1], feature_names=names)   # also: force, decision
plot.bar(exp, x[:50], feature_names=names)        # also: beeswarm (batch-level)
fig.savefig("waterfall.png")                      # everything returns a Figure

# Text — token highlighting for Jupyter plus a static matplotlib export:
out = rot.embed_texts(["a wonderful film", "terrible pacing"])
imp = exp.get_explanation(out.embeddings, mask=out.attention_mask)
plot.text_html(imp[0], out.tokens[0])             # IPython-aware HTML
plot.text_matplotlib(imp[0], out.tokens[0]).savefig("tokens.png")
plot.word_clouds(imp, out.tokens)                 # aggregated pos/neg/combined clouds

# Images — legacy-style saliency overlay (red/blue transparent heatmaps):
plot.saliency(imp_map, image=rgb_array).savefig("saliency.png")
```

Saliency overlays always render at full strength, even when the surrogate
barely tracks the black box — do not read them as detectors (of lesions,
objects, or anything focal). Every fit records `exp.train_agreement_` and
warns below 0.75; check it, and the [Capacity](https://rule-of-thumb.readthedocs.io/en/latest/capacity.html)
guidance, before trusting a map.

**Baseline semantics (SHAP → RoT).** SHAP decomposes `f(x) = φ₀ + Σφᵢ`
with `φ₀ = E[f(X)]`. RoT's surrogate is additive by construction:
`s_k(x) = g_k + Σ_d a_kd·(x_d + b_kd)` and `get_explanation` returns exactly
the per-feature terms — so the SHAP base value maps to the **class bias
`g_k`**, and `f(x)` maps to the **surrogate score** (RoT explains its own
surrogate of the black box, not the black box directly). RoT folds each
feature's shift `b` into its contribution, so features are measured from
zero-shift rather than mean-centred baselines. Text scores are
length-normalised means over tokens, so token importances do not sum to the
score; text visualisations show raw signed token weights only.

## Quick start notebooks

Hello-world examples on dummy data — no downloads or GPUs needed:

- `examples/01_tabular_quickstart.ipynb`
- `examples/02_text_quickstart.ipynb`
- `examples/03_image_quickstart.ipynb`

## Devices

All three RoT models and both explainer wrappers accept `device=`. The
default (`device=None`) auto-detects the best available backend
(CUDA → MPS → CPU). Fit and inference move inputs to the model's device
automatically; every inference entry point (`score`, `predict`,
`ordered_predict`, `score_ordering`, `get_order`, `get_explanation`)
returns host-side results (CPU tensors or numpy arrays) regardless of
`device`, so `np.asarray(exp.predict(X))` works everywhere. Only the
training internals (`importance`, `stochastic_importance`,
`training_loop`) stay on the model's device.

```python
exp = rot.fit(y_outputs=labels, x_inputs=X, device="cuda")  # or "mps", "cpu", ...
```

## Migrating to v0.0.2

v0.0.2 removes the deprecated padding spellings: `lengths=` and
`attention_mask=` kwargs are gone from every entry point — pass a single
boolean validity mask as `mask=` (build it from lengths with
`lengths_to_mask`), and the `sentinel_mask` helper is deleted. New:
`ruleofthumb.vision` (`embed_images` / `ImageEmbeddings` /
`DEFAULT_IMAGE_MODEL`) backs image file paths, `embed_texts` accepts
`revision=` (pinned by default), and `autotune` forwards `n_classes` plus
extra factory kwargs. Save files now load only under the exact package
version that wrote them — refit and re-save after upgrading.

## Migrating from v0.1 sentinel padding

v0.2 removed the implicit `-1` sentinel: **no fill value has special meaning
any more**. Padding is now always explicit via a validity mask (`True` = real
token/pixel). Code changes required:

```python
# v0.1 (implicit): pad with -1, the model inferred padding from the data
x[:, n_tokens:] = -1.0
exp = rot.fit_text(y_outputs=y, x_inputs=x)

# v0.2 (explicit): keep any pad value you like, but pass the mask yourself
x[:, n_tokens:] = 0.0                                # any value works now
mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool)
mask[:, :n_tokens] = True                            # or lengths_to_mask(lengths, T)
exp = rot.fit_text(y_outputs=y, x_inputs=x, mask=mask)
```

Without a mask every position is treated as real data — padded positions are
no longer masked implicitly.

## Migrating from the v0.2.x wrapper classes

v0.2.10 replaced the `RuleOfThumb` / `TextRuleOfThumb` wrapper classes with
one facade: the `Explainer` class plus `fit` / `fit_tabular` / `fit_text` /
`fit_image` factories. Training arguments are unchanged; construction moves
from constructors to factories:

```python
# v0.2.x
from ruleofthumb import RuleOfThumb, TextRuleOfThumb
rot = RuleOfThumb(y_outputs=y, x_inputs=X)
rot = TextRuleOfThumb(y_outputs=y, x_inputs=x, lengths=lengths)

# v0.2.10+
import ruleofthumb as rot
from ruleofthumb.text import lengths_to_mask
exp = rot.fit(y_outputs=y, x_inputs=X)                    # modality auto-detected
exp = rot.fit_text(y_outputs=y, x_inputs=x, mask=lengths_to_mask(lengths, x.shape[1]))
```

The fitted explainer exposes the same `get_explanation` semantics, plus
delegating `get_order` / `ordered_predict` / `score_ordering` / `score` /
`predict` methods (previously reached via the private `_explainer_model`
attribute). The raw models (`RoT`, `RoTText`, `RoTImage`) are unchanged.

Without a mask every position is treated as real data — padded positions are
no longer masked implicitly.

The incremental-reveal pipeline (`get_order` / `ordered_predict` /
`score_ordering`) is mask-aware: padded feature positions are ranked last and
reported as `-1`, and by default reveal curves stop after each sample's real
features are exhausted. By default one reveal step covers a whole **token**
(text) or **pixel** (image) — its embedding dims / channels are revealed
together. Pass `granularity="element"` to `get_order`, `ordered_predict` and
`score_ordering` for the finer per-element curves (the value must match how
the order was produced). Use `include_padded=True` on `ordered_predict` /
`score_ordering` to retain the full rectangular curve including constant
trailing steps. `score_ordering` defaults to per-step accuracy (valid for any
number of classes); pass `return_confusion=True` for per-step K×K confusion
counts (rows = true label, columns = predicted class), or `metric=` for a
custom callable over the binary counts `(tp, fp, fn, tn)`.

## Limitations

- **Rectangular batches.** Inputs are stored as rectangular tensors; use
  `pad_sequences` / `pad_images` plus masks for ragged or mixed-size data.
- Reveal-curve granularity must match between `get_order` and
  `ordered_predict` / `score_ordering` (no auto-detection of the order's
  granularity).
- Custom reveal-curve metrics (`metric=`) are defined over binary counts
  (`tp, fp, fn, tn`); use them only where that view is meaningful. The default
  accuracy metric and `return_confusion=True` work for any number of classes.
- **Pooled linear capacity (text & image).** The text and image variants share
  their importance weights across positions. Definitions: `n_classes` is the
  number of possible answers (2 for cat-vs-dog, 10 for digits 0-9);
  `embedding` (`E`) is the length of the number-list describing one token
  (e.g. 384); `channels` (`C`) is the number of layers per pixel (1 for
  grey, 3 for colour, 576 for MobileNetV3-Small feature maps); `tokens`
  (`T`) is the number of word-pieces per text; `height`/`width` (`H`/`W`)
  are image dimensions. The learned slopes `a` and offsets `b` have shape
  `(n_classes, embedding)` for text and `(n_classes, channels)` for images,
  and the same `a`/`b` row is reused at every token/pixel. Each surrogate is
  therefore a *linear model on a pooled representation*: the token-mean
  embedding (`E` averaged numbers) for text, and per-channel spatial sums
  (`C` totals) for images. Sharing is what makes saliency maps
  position-invariant (where a pattern appears matters less than how much of
  it there is) and keeps the learned count at `2*n_classes*C` independent of
  `T`/`H`/`W`. Fidelity is then bounded by how well classes separate in that
  pooled space. Raw single-channel images (`C = 1`) pool to total ink mass
  alone, capping 10-class digit fidelity near the majority baseline (always
  guessing the commonest class) regardless of training — pinned by
  `test_multiclass_confusion_counts_and_reveal_curve`. Coordinate channels
  (`{mass, row-mass, col-mass}`) lift digits to ~0.35
  (`test_coordinate_channels_restore_multiclass_capacity`); the TinyCNN trunk
  itself (`(500, 8, 8, 8)` maps) reaches only ~0.49 and is not used as a
  backbone. For text, word order is invisible — predictions depend only on
  which words appear — so syntax, negation scope or token counts are
  similarly capped, while lexical sentiment works. In real life prefer rich
  channel representations (e.g. pretrained transformer embeddings or
  MobileNet feature maps): the same shared-weight image surrogate reaches
  ~0.99 on 10-class digits from live MobileNetV3-Small `(500, 576, 7, 7)`
  maps with 500 images exceeding 49 spatial positions — pinned by
  `test_multiclass_rich_backbone_strong_accuracy` (strong bar `>= 0.8`).
  Always check surrogate predicted-class accuracy before trusting an
  explanation. See `ToDo.md` item 16 for the deferred opt-in per-location
  variant.

See `ToDo.md` for the full list and `tests/test_masks.py` for pinned
behaviour.

## Status

v0.0.1 is the first public pre-alpha: a vibe-coded consolidation of the
original experiment code, published to reserve the PyPI name
(`ruleofthumb-rot`) and invite
early feedback. Expect breakage and backwards-incompatible changes before
any 1.0. Internally this continues the 0.2.x line (explicit masks,
unit-granularity reveal curves, explainer facade, native string/path
ingestion, persistence, autotuning, plotting, non-linear shapes); known
limitations and planned follow-ups are tracked in `ToDo.md`.
