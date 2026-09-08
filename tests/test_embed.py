"""Unit tests for :mod:`ruleofthumb.embed` using stubbed tokeniser/model.

The stubs mimic the HuggingFace API surface that ``embed_texts`` relies on
(tokeniser call returning ``input_ids`` / ``attention_mask`` tensors,
``convert_ids_to_tokens`` / ``convert_tokens_to_string``, model forward
returning ``last_hidden_state``) without touching the network.
"""

import numpy as np
import pytest
import torch

from ruleofthumb import DEFAULT_TEXT_MODEL, TextEmbeddings, embed_texts

DIM = 4


class StubTokenizer:
    """Whitespace tokeniser: words get sequential ids (pad id 0), decodable back."""

    def __init__(self):
        self._vocab = {}
        self._inverse = {}

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, max_length=None):
        rows = []
        for text in texts:
            words = text.split()
            if truncation and max_length is not None:
                words = words[:max_length]
            row = []
            for word in words:
                if word not in self._vocab:
                    self._vocab[word] = len(self._vocab) + 1
                    self._inverse[self._vocab[word]] = word
                row.append(self._vocab[word])
            rows.append(row)
        width = max(len(row) for row in rows)
        ids = [row + [0] * (width - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        return [self._inverse.get(i, "[PAD]") for i in ids]

    def convert_tokens_to_string(self, tokens):
        return tokens[0]


class StubOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class StubModel(torch.nn.Module):
    """Deterministic embedding: every dim repeats the token id."""

    def forward(self, input_ids=None, attention_mask=None):
        hidden = input_ids.unsqueeze(-1).float().expand(input_ids.shape[0], input_ids.shape[1], DIM)
        return StubOutput(hidden.clone())


@pytest.fixture()
def stubs():
    return StubTokenizer(), StubModel()


def test_shapes_mask_and_padding_zeros(stubs):
    tokenizer, model = stubs
    out = embed_texts(["a bb ccc", "dddd"], "ignored/model", tokenizer=tokenizer, model=model)

    assert isinstance(out, TextEmbeddings)
    assert out.embeddings.shape == (2, 3, DIM)
    assert out.embeddings.dtype == np.float32
    assert out.attention_mask.shape == (2, 3)
    assert out.attention_mask.dtype == bool
    assert out.attention_mask.tolist() == [[True, True, True], [True, False, False]]
    # real-token embeddings repeat the token id; padded rows are exactly zero
    assert np.array_equal(out.embeddings[0, :, 0], [1.0, 2.0, 3.0])
    assert np.array_equal(out.embeddings[1, 0, 0], 4.0)
    assert (out.embeddings[1, 1:] == 0).all()


def test_tokens_align_with_mask(stubs):
    tokenizer, model = stubs
    out = embed_texts(["a bb ccc", "dddd"], tokenizer=tokenizer, model=model)
    assert out.tokens[0] == ["a", "bb", "ccc"]
    assert out.tokens[1] == ["dddd", "", ""]


def test_batch_size_invariance(stubs):
    tokenizer, model = stubs
    texts = ["a", "bb ccc", "dddd ee", "fff"]
    one = embed_texts(texts, tokenizer=tokenizer, model=model, batch_size=1)
    many = embed_texts(texts, tokenizer=tokenizer, model=model, batch_size=4)
    assert np.array_equal(one.embeddings, many.embeddings)
    assert np.array_equal(one.attention_mask, many.attention_mask)
    # token rows are padded to the batch-wide max length; compare real tokens only
    assert [[t for t in row if t] for row in one.tokens] == [[t for t in row if t] for row in many.tokens]


def test_max_length_truncates(stubs):
    tokenizer, model = stubs
    out = embed_texts(["a bb ccc dddd eeeee"], tokenizer=tokenizer, model=model, max_length=3)
    assert out.embeddings.shape == (1, 3, DIM)
    assert out.tokens[0] == ["a", "bb", "ccc"]


def test_tokenizer_model_override_ignores_model_name(stubs):
    tokenizer, model = stubs
    out = embed_texts(["a bb"], "this/model-does-not-exist", tokenizer=tokenizer, model=model)
    assert out.embeddings.shape == (1, 2, DIM)


def test_device_argument(stubs):
    tokenizer, model = stubs
    out = embed_texts(["a bb"], tokenizer=tokenizer, model=model, device="cpu")
    assert out.embeddings.shape == (1, 2, DIM)


def test_partial_override_rejected(stubs):
    with pytest.raises(ValueError, match="both"):
        embed_texts(["a"], tokenizer=StubTokenizer())


def test_empty_texts_rejected(stubs):
    tokenizer, model = stubs
    with pytest.raises(ValueError, match="non-empty"):
        embed_texts([], tokenizer=tokenizer, model=model)


def test_bad_batch_size_rejected(stubs):
    tokenizer, model = stubs
    with pytest.raises(ValueError, match="batch_size"):
        embed_texts(["a"], tokenizer=tokenizer, model=model, batch_size=0)


def test_default_model_constant():
    assert DEFAULT_TEXT_MODEL == "answerdotai/ModernBERT-base"


def test_fit_routes_strings_to_text(stubs):
    from ruleofthumb import fit

    tokenizer, model = stubs
    exp = fit(np.array([0, 1]), ["a bb", "ccc"], tokenizer=tokenizer, model=model, epochs=2, batch_size=2, learning_rate=0.05)
    assert exp.modality == "text"
    assert exp.get_explanation(["a bb", "ccc"]).shape == (2, 2)


def test_facade_methods_accept_strings(stubs):
    from ruleofthumb import fit_text

    tokenizer, model = stubs
    y = np.array([0, 1, 0])
    texts = ["a bb", "ccc", "dddd ee"]
    exp = fit_text(y, texts, tokenizer=tokenizer, model=model, epochs=2, batch_size=3, learning_rate=0.05)

    imp = exp.get_explanation(texts)
    assert imp.shape == (3, 2)  # padded to the longest text ("a bb" / "dddd ee" = 2 tokens)
    order = exp.get_order(texts)
    assert order.shape == imp.shape
    curve = exp.score_ordering(texts, y, order)
    assert curve.ndim == 1 and len(curve) > 0
    preds = exp.predict(texts)
    assert preds.shape[0] == 3


def test_strings_with_explicit_padding_rejected(stubs):
    from ruleofthumb import fit_text

    tokenizer, model = stubs
    with pytest.raises(ValueError, match="automatically"):
        fit_text(np.array([0, 1]), ["a bb", "ccc"], attention_mask=np.ones((2, 3)), tokenizer=tokenizer, model=model)


def test_get_explanation_strings_on_array_fitted_raises(stubs):
    from ruleofthumb import fit_text

    x = np.random.rand(4, 3, DIM).astype(np.float32)
    exp = fit_text(np.array([0, 1, 0, 1]), x, epochs=2, batch_size=4, learning_rate=0.05)
    with pytest.raises(ValueError, match="fitted on embedding arrays"):
        exp.get_explanation(["a bb", "ccc"])
