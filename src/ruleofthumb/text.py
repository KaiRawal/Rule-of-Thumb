"""Text RoT variant.

Port of ``RoT_text`` from the original experiment code
(``ExplanationExampleRemote/rot_class.py``). Assumes datapoints are in the
form: token x embedding.

Since v0.2 padding is explicit: every method takes an optional boolean
``mask`` of shape ``(N, tokens)`` marking real (non-padding) tokens. There is
no implicit ``-1`` sentinel detection any more; use :func:`sentinel_mask` to
migrate legacy ``-1``-padded arrays or :func:`pad_sequences` to pad ragged
inputs.
"""

import torch
from torch import nn

from ruleofthumb.core import RoT


def lengths_to_mask(lengths, max_len):
    """Convert per-sequence lengths to a ``(N, max_len)`` boolean mask."""
    lengths = torch.as_tensor(lengths)
    return (torch.arange(max_len)[None, :] < lengths[:, None]).to(torch.bool)


def pad_sequences(sequences, pad_value=0.0):
    """Pad a list of ``(T_i, E)`` sequences into one rectangular batch.

    Returns ``(padded, lengths)`` where ``padded`` has shape
    ``(N, max(T_i), E)`` padded with ``pad_value``, and ``lengths`` holds each
    sequence's true token count. Pass the lengths/mask on to the model so
    padded positions are ignored.
    """
    n = len(sequences)
    seqs = [torch.as_tensor(s) for s in sequences]
    max_len = int(max(s.shape[0] for s in seqs))
    embedding = int(seqs[0].shape[1])
    padded = torch.full((n, max_len, embedding), float(pad_value), dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s
    return padded, torch.tensor([int(s.shape[0]) for s in seqs])


def sentinel_mask(points, value=-1.0):
    """Rebuild an explicit validity mask from legacy sentinel-padded data.

    v0.1 treated tokens whose embedding was entirely ``value`` (by default
    ``-1``) as padding. New code should pass masks explicitly; this helper
    reproduces that detection for migrating old inputs. Returns a validity
    mask (``True`` marks real tokens).
    """
    points = torch.as_tensor(points)
    return ~(points == value).all(dim=-1)


class RoTText(RoT):
    """Token x embedding RoT explainer.

    Definitions: `E` is the embedding length per token (numbers describing
    one word-piece); the learned slopes `a` and offsets `b` have one row per
    embedding number and that row is reused at every token, so the score is
    linear on the token-mean embedding. Word order is therefore invisible:
    lexical tasks work, syntax/negation/count tasks are capped (see README
    Limitations). Rich pretrained embeddings are the intended real-life
    input.

    All methods accept an optional ``mask`` of shape ``(N, tokens)``:
    ``True``/1 marks real tokens, masked-out tokens receive exactly zero
    importance and are excluded from length normalisation. With ``mask=None``
    every token is treated as real data.
    """

    def __init__(self, classes, sample_shape, dropout_rate=0.5, use_BCE_loss=False, l1_penalty=0.01, device=None, nonlinear=None):
        super().__init__(classes, sample_shape, dropout_rate, use_BCE_loss, no_a_b=True, device=device, nonlinear=nonlinear)
        self.a = nn.Parameter(torch.zeros((classes, sample_shape[1]), requires_grad=True, device=self.device))
        self.b = nn.Parameter(torch.zeros((classes, sample_shape[1]), requires_grad=True, device=self.device))
        self.weights = (self.a, self.b, self.g)
        self.l1_penalty = l1_penalty

    def importance(self, points, mask=None):
        points = torch.as_tensor(points, device=self.device)
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
        imp = self.a[None, :, None, :] * (self._respond(points)[:, None] + self.b[None, :, None, :])
        if mask is None:
            return imp
        return mask[:, None, :, None].to(imp.dtype) * imp

    def stochastic_importance(self, points, mask=None):
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
        imp = self.importance(points, mask=mask)
        token_dim = 1
        keep = (
            torch.rand(points.shape[0], points.shape[token_dim], device=self.device) > self.dropout_rate
        ).float()
        if mask is not None:
            keep = keep * mask.to(keep.dtype)
        return keep[:, None, :, None] * imp

    def score(self, points, mask=None):
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
        imp = self.importance(points, mask=mask).detach()
        response_sum = imp.sum(dim=2)
        if mask is None:
            length = torch.full(
                response_sum.shape[:1], points.shape[1], dtype=response_sum.dtype, device=self.device
            )
        else:
            length = mask.to(response_sum.dtype).sum(1)
        response_mean = response_sum / length.clamp(min=1)[:, None, None]

        score = response_mean.reshape(response_mean.shape[0], response_mean.shape[1], -1).sum(-1)
        score += self.g[None, :]
        return score

    def loss(self, points, target, mask=None):
        response = self.stochastic_importance(points, mask=mask)
        response = response.reshape(points.shape[0], self.classes, -1).sum(-1)
        response = response + self.g[None]
        base_loss = self.objective(response, target)

        l1_loss = (
            self.l1_penalty
            * points.shape[0]
            * self.classes
            * (self.a.abs().sum() + self.b.abs().sum() + self.g.abs().sum())
        )
        return base_loss + l1_loss

    def _reduce_to_units(self, imp):
        """Reveal units are tokens: aggregate signed importance over embedding dims."""
        return imp.sum(dim=-1)

    def fit(
        self,
        points,
        classifier_response,
        epochs,
        batch_size,
        lr=1e-4,
        mask=None,
        pretrain_epochs=5,
        weight_decay=0.01,
        seed=None,
    ):
        # Unlike the tabular ``RoT.fit``: no projection of ``b`` onto the data
        # range, zero initialisation instead of mean-centering, as in the
        # original text-experiment copies.
        points = torch.as_tensor(points, device=self.device)
        classifier_response = torch.as_tensor(classifier_response, device=self.device)
        if mask is not None:
            mask = torch.as_tensor(mask, device=self.device)
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            self.b.zero_()
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr)
        drop_out = self.dropout_rate
        self.dropout_rate = 0
        self.training_loop(self.loss, points, classifier_response, optimiser, pretrain_epochs, batch_size, mask=mask)
        self.dropout_rate = drop_out
        optimiser = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.training_loop(self.loss, points, classifier_response, optimiser, epochs, batch_size, mask=mask)
