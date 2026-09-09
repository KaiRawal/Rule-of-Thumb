"""Embedding-extraction utilities for text.

Turns raw strings into the ``(N, tokens, embedding)`` arrays (plus validity
masks and decoded token strings) that :class:`ruleofthumb.text.RoTText`
consumes, so callers never hand-build them. Port of the legacy
``gen_token_embeddings.py`` workflow as a library function instead of a
script.

The default model is ``answerdotai/ModernBERT-base``; both the tokeniser and
the model can be overridden by callers who need a different backbone.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ruleofthumb.core import _resolve_device

DEFAULT_TEXT_MODEL = "answerdotai/ModernBERT-base"

#: Pinned HuggingFace revision of :data:`DEFAULT_TEXT_MODEL`, so embeddings
#: reproduce exactly on every machine (CI cold-downloads match warm caches).
DEFAULT_TEXT_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


@dataclasses.dataclass(frozen=True)
class TextEmbeddings:
    """Token embeddings for a batch of texts, ready for ``fit_text``.

    Attributes:
        embeddings: ``(N, tokens, dim)`` float32 array, zero-padded beyond
            each sample's true length.
        attention_mask: ``(N, tokens)`` boolean validity mask (``True`` marks
            real tokens); pass as ``mask=`` to ``fit_text``.
        tokens: per-sample list of decoded token strings aligned with the
            embedding rows; padding positions are empty strings.
    """

    embeddings: np.ndarray
    attention_mask: np.ndarray
    tokens: list


def _resolve_tokenizer_model(tokenizer, model, model_name, revision):
    if (tokenizer is None) != (model is None):
        raise ValueError("pass both tokenizer= and model=, or neither")
    if tokenizer is not None:
        return tokenizer, model
    from transformers import AutoModel, AutoTokenizer

    return (
        AutoTokenizer.from_pretrained(model_name, revision=revision),
        AutoModel.from_pretrained(model_name, revision=revision),
    )


def _decode_tokens(tokenizer, input_ids):
    return [tokenizer.convert_tokens_to_string([t]).strip() for t in tokenizer.convert_ids_to_tokens(input_ids)]


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_TEXT_MODEL,
    *,
    revision: str | None = DEFAULT_TEXT_REVISION,
    tokenizer: Any | None = None,
    model: Any | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
    device: Any | None = None,
) -> TextEmbeddings:
    """Tokenise and embed raw strings with a HuggingFace transformer.

    Args:
        texts: list of raw strings.
        model_name: HuggingFace model used when ``tokenizer``/``model`` are
            not supplied.
        revision: pinned HuggingFace revision for ``model_name`` (defaults
            to :data:`DEFAULT_TEXT_REVISION`); pass ``None`` for the hub
            default. Ignored when ``tokenizer``/``model`` are supplied.
        tokenizer: optional pre-loaded tokeniser; supplying it (together with
            ``model``) makes ``model_name`` irrelevant.
        model: optional pre-loaded ``AutoModel`` in eval mode (it is put into
            eval mode here regardless).
        max_length: optional truncation length forwarded to the tokeniser.
        batch_size: number of texts encoded per forward pass.
        device: torch device for the forward pass; ``None`` auto-detects
            cuda > mps > cpu. Outputs are always returned on CPU.

    Returns:
        :class:`TextEmbeddings` with rectangular zero-padded embeddings, a
        boolean attention mask and decoded token strings.
    """
    if len(texts) == 0:
        raise ValueError("texts must be non-empty")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    tokenizer, model = _resolve_tokenizer_model(tokenizer, model, model_name, revision)
    device = _resolve_device(device)
    model = model.to(device).eval()

    encode_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
    if max_length is not None:
        encode_kwargs["max_length"] = max_length

    chunk_masks, chunk_embeddings, chunk_tokens = [], [], []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start : start + batch_size], **encode_kwargs)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        chunk_masks.append(encoded["attention_mask"].cpu())
        chunk_embeddings.append(hidden.cpu())
        chunk_tokens.extend(_decode_tokens(tokenizer, ids.tolist()) for ids in encoded["input_ids"].cpu())

    n = len(texts)
    max_len = int(max(chunk.shape[1] for chunk in chunk_embeddings))
    dim = int(chunk_embeddings[0].shape[2])
    embeddings = np.zeros((n, max_len, dim), dtype=np.float32)
    attention_mask = np.zeros((n, max_len), dtype=bool)
    idx = 0
    for emb_chunk, mask_chunk in zip(chunk_embeddings, chunk_masks):
        length = emb_chunk.shape[1]
        count = mask_chunk.shape[0]
        embeddings[idx : idx + count, :length] = emb_chunk.numpy()
        chunk_mask = mask_chunk.numpy().astype(bool)
        attention_mask[idx : idx + count, :length] = chunk_mask
        for j in range(count):
            tokens = chunk_tokens[idx + j]
            tokens += [""] * (max_len - len(tokens))
            chunk_tokens[idx + j] = [t if real else "" for t, real in zip(tokens, chunk_mask[j])]
        idx += count
    # transformers emit non-zero hidden states at padded positions; enforce the
    # package-wide padding contract (the mask carries the truth) explicitly
    embeddings[~attention_mask] = 0.0
    return TextEmbeddings(embeddings=embeddings, attention_mask=attention_mask, tokens=chunk_tokens)
