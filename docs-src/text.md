# Text

Sentences in, one importance number per word out. This is where RoT
arguably earns its keep: the language models most worth explaining
are so often closed APIs, and this page starts from raw strings, so
padding and masks are quietly handled for you:

```python
import ruleofthumb as rot

texts = ["a wonderful film", "terrible pacing"]
exp = rot.fit_text(y_answers, texts)
imp = exp.get_explanation(texts)  # [2, words], signed; filler scores 0
order = exp.get_order(texts)      # best word first per sentence
```

If you later want the intermediate pieces — the decoded words for
plotting, say — just ask for them explicitly:

```python
out = rot.embed_texts(texts)  # embeddings, mask, tokens
exp = rot.fit_text(y_answers, out.embeddings, mask=out.attention_mask)
```

## When you already hold the arrays

Sometimes the embeddings are already sitting in your workspace as a
rectangular `[N, tokens, dim]` array. Ragged lists become rectangular
with `pad_sequences` (any filler value will do), and the mask follows
from the lengths — then pass both everywhere:

```python
from ruleofthumb.text import lengths_to_mask, pad_sequences

x, lengths = pad_sequences(sequences)               # [N, T, E]
mask = lengths_to_mask(lengths, x.shape[1]).numpy() # [N, T], True = real
exp = rot.fit_text(y_answers, x.numpy(), mask=mask)
imp = exp.get_explanation(x.numpy(), mask=mask)     # [N, T]
```

A few rules that repay memorising: `mask=True` marks a genuine word;
filler positions score exactly zero and rank last (as `-1` in
orders); your tokenizer's own `attention_mask` works as-is; and
`score_ordering` takes no mask at all, because the `-1`s in the order
already carry that information. [Core ideas](concepts.md) has the
picture if the words alone do not land.

## Revealing and drawing words

One reveal step always covers a whole **word** — its embedding
dimensions travel together. The mechanics are the same as everywhere
else (see [Checking the ranking by revealing less](reveal.md)):

```python
fig = plot.text_matplotlib(imp[0], out.tokens[0])  # static export
plot.text_html(imp[0], out.tokens[0])              # notebook highlight
fig = plot.word_clouds(imp, out.tokens)            # batch clouds
```

The full gallery is under [Plots](plots.md). For runnable versions,
try the [Text demo](notebooks/02_text_quickstart.ipynb) first and
keep [HateXPlain](notebooks/04_hatexplain.ipynb) for when you are
curious how explanations compare with human judgements.

## One caution before you trust it

The stand-in sees the *average* of a sentence's words rather than
their order, so "dog bites man" and "man bites dog" receive the same
explanation. Tasks driven by *which* words appear — sentiment, topic,
screening — play to this strength; syntax and negation scope do not.
[Limits](capacity.md) says more.
