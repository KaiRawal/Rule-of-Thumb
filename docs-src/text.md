# Text guide

Inputs are `(N, tokens, embedding)` float arrays. Padding is explicit and
first-class: pass a boolean validity `mask` (`True` = real token) — no fill
value has special meaning.

## Embeddings + masks

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
# signed, shape [N, max_tokens]; padded tokens score exactly 0
# (for n_classes > 2 the output is per-class instead: [N, n_classes, max_tokens])
```

Already have a rectangular batch and your own mask? Pass it directly as
`mask=` to `fit_text` / `get_explanation` / `get_order` — HuggingFace
tokenizer `attention_mask` tensors compose as-is. `score_ordering` takes no
mask: padding is already encoded as `-1` entries in the order.

## Raw strings

`fit_text` embeds strings itself (bundled default
`answerdotai/ModernBERT-base`, overridable via `tokenizer=` / `model=`),
derives padding automatically, and every explainer method accepts the same
strings back:

```python
import ruleofthumb as rot

exp = rot.fit_text(y_outputs=labels, x_inputs=["a wonderful film", "terrible pacing"])
token_importances = exp.get_explanation(["a wonderful film", "terrible pacing"])
order = exp.get_order(["a wonderful film", "terrible pacing"])
```

Need the intermediate arrays (e.g. decoded tokens for plotting)? Use
`rot.embed_texts` directly:

```python
out = rot.embed_texts(["a wonderful film", "terrible pacing"])
exp = rot.fit_text(y_outputs=labels, x_inputs=out.embeddings,
                           mask=out.attention_mask)
out.tokens  # decoded token strings, aligned with per-token importances
```

## Reveal pipeline

One reveal step covers a whole **token** (embedding dims revealed together).
Pass `granularity="element"` to `get_order`, `ordered_predict` and
`score_ordering` for per-element curves — the value must match how the
order was produced. Details: {ref}`Workflows: reveal curves <reveal-curves>`.

## Worked notebook

The executed hello-world is
`notebooks/02_text_quickstart.ipynb` (generated copy of
`examples/02_text_quickstart.ipynb`, re-executed on every site build).

## Next steps

- Word order is invisible to the surrogate (token-mean pooling) — read
  [Capacity](capacity.md) before trusting an explanation.
- Token plots and word clouds: {ref}`Workflows: plotting <plotting>`.
