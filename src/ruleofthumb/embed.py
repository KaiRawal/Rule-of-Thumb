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
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as _exc:
        raise ImportError("embed_texts needs transformers: pip install ruleofthumb-rot[text]") from _exc

    return (
        AutoTokenizer.from_pretrained(model_name, revision=revision),
        AutoModel.from_pretrained(model_name, revision=revision),
    )


def _decode_tokens(tokenizer, input_ids):
    return [tokenizer.convert_tokens_to_string([t]).strip() for t in tokenizer.convert_ids_to_tokens(input_ids)]


def _embed_strided(tokenizer, model, text, window, stride, device):
    """Embed one text in overlapping windows; returns ``(embeddings, input_ids)`` in the usual layout.

    Each window holds ``window`` text tokens wrapped once in the special tokens. A text token keeps its row
    from the first window that contains it; the leading / trailing special tokens come from the first / last window.
    """
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    front = tokenizer.build_inputs_with_special_tokens([-1]).index(-1)  # special tokens before the text
    parts, done = [], 0
    for start in range(0, max(len(ids), 1), stride):
        chunk = ids[start : start + window]
        inputs = torch.tensor([tokenizer.build_inputs_with_special_tokens(chunk)], device=device)
        with torch.no_grad():
            hidden = model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).last_hidden_state[0].cpu()
        last = start + len(chunk) == len(ids)
        parts.append(hidden[0 if start == 0 else front + done - start : None if last else front + len(chunk)])
        done = start + len(chunk)
        if last:
            break
    return torch.cat(parts), tokenizer.build_inputs_with_special_tokens(ids)


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_TEXT_MODEL,
    *,
    revision: str | None = DEFAULT_TEXT_REVISION,
    tokenizer: Any | None = None,
    model: Any | None = None,
    max_length: int | None = None,
    stride: int | None = None,
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
            With ``stride`` it is instead the model's window size (special
            tokens included, e.g. 512), and nothing is truncated.
        stride: embed long texts in overlapping windows instead of truncating
            them. Windows hold ``max_length`` minus the special tokens and start
            ``stride`` tokens apart; each token keeps its embedding from the
            first window that contains it. Legacy Judicial recipe:
            ``max_length=512, stride=410`` (RoBERTa: 510-token windows, overlap
            100), minus its doubled ``<s>`` / ``</s>``. Requires ``max_length``;
            ``batch_size`` is ignored.
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
    if stride is not None:
        if max_length is None:
            raise ValueError("stride= needs max_length=, the model's window size (e.g. 512)")
        window = max_length - tokenizer.num_special_tokens_to_add()
        if not 0 < stride <= window:
            raise ValueError(f"stride must be in 1..{window} (the window without special tokens), got {stride}")
    device = _resolve_device(device)
    model = model.to(device).eval()

    encode_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
    if max_length is not None:
        encode_kwargs["max_length"] = max_length

    chunk_masks, chunk_embeddings, chunk_tokens = [], [], []
    if stride is not None:
        for text in texts:
            hidden, ids = _embed_strided(tokenizer, model, text, window, stride, device)
            chunk_masks.append(torch.ones(1, len(ids), dtype=torch.long))
            chunk_embeddings.append(hidden[None])
            chunk_tokens.append(_decode_tokens(tokenizer, ids))
    else:
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
