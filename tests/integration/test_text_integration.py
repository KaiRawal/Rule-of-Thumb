"""Integration tests: text modality against a real transformer.

The black box is the cached ``distilbert-base-uncased-finetuned-sst-2-english``
model; its token embeddings and predicted labels are computed live at session
start. Only the RoT explainer is fitted, through the
:func:`ruleofthumb.fit_text` facade.
"""

import re

import numpy as np
import pytest
import torch
from _helpers import rot_accuracy

from ruleofthumb import fit_text

SEED = 0

POSITIVE_WORDS = {
    "brilliant", "wonderful", "best", "superb", "delightful", "phenomenal",
    "masterpiece", "gem", "gorgeous", "recommended", "loved", "excellent",
    "charming", "tender", "haunting", "vibrant", "inventive", "riveting", "graceful",
}
NEGATIVE_WORDS = {
    "bored", "boring", "dull", "terrible", "worst", "awful", "waste", "forgettable",
    "disappointment", "avoid", "chore", "mess", "wooden", "lifeless", "mediocre",
    "unfunny", "insults", "cheap", "clumsy", "lazy",
}


def _iter_words(encoded, tokenizer, i):
    """Yield ``(word_id, token_positions, word)`` for every real word in review *i*."""
    tokens = encoded.tokens(i)
    groups = {}
    for position, word_id in enumerate(encoded.word_ids(i)):
        if word_id is None:  # [CLS] / [SEP] / padding
            continue
        groups.setdefault(word_id, []).append(position)
    for word_id, positions in groups.items():
        raw = tokenizer.convert_tokens_to_string([tokens[p] for p in positions])
        yield word_id, positions, re.sub(r"[^a-z]", "", raw.lower())


def _fit_text(embeddings, attention_mask, y):
    return fit_text(y, embeddings, attention_mask=attention_mask, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)


def test_black_box_predictions_are_confident(text_sst2):
    assert text_sst2["n_classes"] == 2
    # the fixed reviews must be clearly polarised so sign checks are meaningful
    assert float(text_sst2["confidence"].min()) >= 0.9


def test_explanation_shape_and_padding_zeros(text_sst2):
    embeddings = text_sst2["embeddings"]
    mask = text_sst2["attention_mask"]
    exp = _fit_text(embeddings, mask.numpy(), text_sst2["y"])

    imp = exp.get_explanation(embeddings, attention_mask=mask.numpy())
    n, tokens = mask.shape
    assert imp.shape == (n, tokens)
    assert np.isfinite(imp).all()
    # padded tokens receive exactly zero importance
    assert (imp[~mask.numpy()] == 0).all()
    assert (np.abs(imp) > 0).any()


def test_sentiment_words_carry_signed_importance(text_sst2):
    tokenizer = text_sst2["tokenizer"]
    encoded = text_sst2["encoded"]
    exp = _fit_text(text_sst2["embeddings"], text_sst2["attention_mask"].numpy(), text_sst2["y"])
    imp = exp.get_explanation(text_sst2["embeddings"], attention_mask=text_sst2["attention_mask"].numpy())

    positive_mass, negative_mass = 0.0, 0.0
    for i in range(len(text_sst2["texts"])):
        for _, positions, word in _iter_words(encoded, tokenizer, i):
            mass = float(imp[i, positions].sum())
            if word in POSITIVE_WORDS:
                positive_mass += mass
            elif word in NEGATIVE_WORDS:
                negative_mass += mass

    # class-1 contributions: positive words push toward "positive", negative away
    assert positive_mass > 0.0
    assert negative_mass < 0.0


def test_top_tokens_carry_sentiment_words(text_sst2):
    """Explicit importance check: sentiment words dominate each review's top tokens."""
    encoded, tokenizer = text_sst2["encoded"], text_sst2["tokenizer"]
    exp = _fit_text(text_sst2["embeddings"], text_sst2["attention_mask"].numpy(), text_sst2["y"])
    imp = exp.get_explanation(text_sst2["embeddings"], attention_mask=text_sst2["attention_mask"].numpy())

    positive_hits = negative_hits = positive_n = negative_n = 0
    for i, text in enumerate(text_sst2["texts"]):
        words = {word: abs(float(imp[i, positions].sum())) for _, positions, word in _iter_words(encoded, tokenizer, i)}
        top5 = set(sorted(words, key=words.get, reverse=True)[:5])
        if text_sst2["y"][i] == 1:
            positive_n += 1
            positive_hits += bool(top5 & POSITIVE_WORDS)
        else:
            negative_n += 1
            negative_hits += bool(top5 & NEGATIVE_WORDS)

    assert positive_hits / positive_n >= 0.4
    assert negative_hits / negative_n >= 0.55


def test_brilliant_outranks_awful_in_the_pair_review(text_sst2):
    """In the mixed review containing both words, 'brilliant' must win on class-1 mass."""
    encoded, tokenizer = text_sst2["encoded"], text_sst2["tokenizer"]
    exp = _fit_text(text_sst2["embeddings"], text_sst2["attention_mask"].numpy(), text_sst2["y"])
    imp = exp.get_explanation(text_sst2["embeddings"], attention_mask=text_sst2["attention_mask"].numpy())

    pair_rows = [
        i
        for i in range(len(text_sst2["texts"]))
        if {"brilliant", "awful"} <= {word for _, _, word in _iter_words(encoded, tokenizer, i)}
    ]
    assert len(pair_rows) >= 1
    for i in pair_rows:
        masses = {
            word: float(imp[i, positions].sum())
            for _, positions, word in _iter_words(encoded, tokenizer, i)
            if word in ("brilliant", "awful")
        }
        assert text_sst2["y"][i] == 1  # the review reads as positive overall
        assert masses["brilliant"] > masses["awful"]


def test_reveal_curve_recovers_full_accuracy(text_sst2):
    embeddings, y = text_sst2["embeddings"], text_sst2["y"]
    mask = text_sst2["attention_mask"].numpy()
    exp = _fit_text(embeddings, mask, y)

    xt = torch.from_numpy(embeddings)
    order = exp.get_order(xt, mask=torch.from_numpy(mask))
    # padded positions ranked last and reported as -1
    lengths = text_sst2["attention_mask"].sum(1)
    for i, length in enumerate(lengths):
        assert (order[i, :length] != -1).all() and (order[i, length:] == -1).all()

    curve = exp.score_ordering(xt, torch.from_numpy(y.astype(np.int64)), order)
    # RoT predicted-class accuracy vs the transformer's labels
    full_accuracy = rot_accuracy(exp, embeddings, y)
    assert abs(float(curve[-1]) - full_accuracy) < 1e-6
    assert full_accuracy >= 0.85


def test_seed_reproducibility(text_sst2):
    embeddings, y = text_sst2["embeddings"], text_sst2["y"]
    mask = text_sst2["attention_mask"].numpy()
    imp_a = _fit_text(embeddings, mask, y).get_explanation(embeddings, attention_mask=mask)
    imp_b = _fit_text(embeddings, mask, y).get_explanation(embeddings, attention_mask=mask)
    assert np.allclose(imp_a, imp_b)


def test_embed_texts_produces_fit_ready_arrays():
    """The bundled ModernBERT path yields rectangular embeddings, masks and tokens."""
    from ruleofthumb import DEFAULT_TEXT_MODEL, embed_texts

    texts = ["a wonderful masterpiece", "terrible"]
    out = embed_texts(texts, DEFAULT_TEXT_MODEL, batch_size=2)

    n, tokens, dim = out.embeddings.shape
    assert (n, dim) == (2, 768)
    assert out.attention_mask.shape == (n, tokens)
    assert len(out.tokens) == n and all(len(row) == tokens for row in out.tokens)
    # every sample keeps its words plus [CLS]/[SEP]; pads are blanked everywhere
    assert (out.attention_mask.sum(1) >= [4, 3]).all()
    assert all(token != "" for row, mask in zip(out.tokens, out.attention_mask) for token, real in zip(row, mask) if real)
    assert all(token == "" for row, mask in zip(out.tokens, out.attention_mask) for token, real in zip(row, mask) if not real)
    assert np.isfinite(out.embeddings).all()
    assert (out.embeddings[~out.attention_mask] == 0).all()
    assert (np.abs(out.embeddings[out.attention_mask]) > 0).any()


def test_native_string_ingestion_end_to_end(text_sst2):
    """Raw strings + black-box predictions in; per-token importances out.

    No ``(N, tokens, embedding)`` array is ever built by the caller: fit_text
    embeds via the bundled ModernBERT default, derives the attention mask,
    and get_explanation re-embeds the same strings.
    """
    from ruleofthumb import DEFAULT_TEXT_MODEL, embed_texts

    texts, y = text_sst2["texts"], text_sst2["y"]
    exp = fit_text(y, texts, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)

    embedded = embed_texts(texts, DEFAULT_TEXT_MODEL)
    imp = exp.get_explanation(texts)

    assert exp.modality == "text"
    assert imp.shape == (len(texts), embedded.embeddings.shape[1])
    assert np.isfinite(imp).all()
    assert (imp[~embedded.attention_mask] == 0).all()  # derived pads score exactly zero
    assert (np.abs(imp) > 0).any()

    # RoT predicted-class accuracy against the black box's labels
    assert rot_accuracy(exp, embedded.embeddings, y) >= 0.8

    # signed sentiment-word masses over the ModernBERT tokenisation
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(DEFAULT_TEXT_MODEL)
    encoded = tokenizer(list(texts))
    positive_mass, negative_mass = 0.0, 0.0
    for i in range(len(texts)):
        for _, positions, word in _iter_words(encoded, tokenizer, i):
            mass = float(imp[i, positions].sum())
            if word in POSITIVE_WORDS:
                positive_mass += mass
            elif word in NEGATIVE_WORDS:
                negative_mass += mass
    assert positive_mass > 0.0
    assert negative_mass < 0.0


def test_native_path_equals_array_path(text_sst2):
    """Fitting from strings with an explicit override matches the array path exactly."""
    transformers = pytest.importorskip("transformers")
    from ruleofthumb import embed_texts

    tokenizer = transformers.AutoTokenizer.from_pretrained(text_sst2["model_name"])
    backbone = transformers.AutoModel.from_pretrained(text_sst2["model_name"])
    backbone.eval()

    texts, y = text_sst2["texts"], text_sst2["y"]
    emb = embed_texts(texts, tokenizer=tokenizer, model=backbone, max_length=48)

    native = fit_text(y, texts, tokenizer=tokenizer, model=backbone, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED)
    arrays = fit_text(
        y, emb.embeddings, attention_mask=emb.attention_mask, epochs=200, batch_size=500, learning_rate=0.05, seed=SEED
    )

    imp_native = native.get_explanation(texts)
    imp_arrays = arrays.get_explanation(emb.embeddings, attention_mask=emb.attention_mask)
    assert imp_native.shape == imp_arrays.shape
    assert np.allclose(imp_native, imp_arrays)
