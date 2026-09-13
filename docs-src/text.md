# Text

Sentences in, one importance number per word out. Start from raw
strings — padding and masks are handled for you:

```python
import ruleofthumb as rot

texts = ["a wonderful film", "terrible pacing"]
exp = rot.fit_text(y_answers, texts)
imp = exp.get_explanation(texts)  # [2, words], signed; filler scores 0
order = exp.get_order(texts)      # best word first per sentence
```

Need the intermediate pieces (decoded words for plotting)? Ask for
them explicitly:

```python
out = rot.embed_texts(texts)  # embeddings, mask, tokens
exp = rot.fit_text(y_answers, out.embeddings, mask=out.attention_mask)
```

## Power mode: arrays and masks

Already have token embeddings as a rectangular `[N, tokens, dim]`
array? Pad ragged lists with `pad_sequences` (any filler works) and
build the mask from lengths — then pass both everywhere:

```python
from ruleofthumb.text import lengths_to_mask, pad_sequences

x, lengths = pad_sequences(sequences)               # [N, T, E]
mask = lengths_to_mask(lengths, x.shape[1]).numpy() # [N, T], True = real
exp = rot.fit_text(y_answers, x.numpy(), mask=mask)
imp = exp.get_explanation(x.numpy(), mask=mask)     # [N, T]
```

Rules: `mask=True` means a real word; filler positions score exactly
zero and rank last (`-1` in orders). Your tokenizer's `attention_mask`
works as-is. `score_ordering` takes no mask — the `-1`s in the order
already encode it. See [Core ideas](concepts.md) for the picture.

## Reveal and draw

One reveal step covers a whole **word** (its embedding dims go
together). Details: [Reveal curves](reveal.md).

```python
fig = plot.text_matplotlib(imp[0], out.tokens[0])  # static export
plot.text_html(imp[0], out.tokens[0])              # notebook highlight
fig = plot.word_clouds(imp, out.tokens)            # batch clouds
```

More: [Plots](plots.md). The executed hello-worlds are
[Text demo](notebooks/02_text_quickstart.ipynb) and the
[HateXPlain](notebooks/04_hatexplain.ipynb) reading.

## A note before trusting

The stand-in sees the *average* of a sentence's words, not their
order — "dog bites man" and "man bites dog" get the same explanation.
Lexical tasks (sentiment) work well; syntax and negation do not. See
[Limits](capacity.md).
