"""Automatic hyperparameter tuning for RoT explainers.

Keras-tuner-style search over the exposed training hyperparameters
(``learning_rate``, ``batch_size``, ``epochs``, ``weight_decay``, plus
``l1_penalty`` for text): candidates are fitted on a seeded train split,
scored on held-out data (by plain agreement unless ``scoring="reveal"`` is
named), and the winner is refit on all data. The default search space is
per-modality (:data:`DEFAULT_SPACES`) so each one reaches its own working
regime. With four or more candidates the search screens them cheaply and finishes only
the better half (successive halving). A candidate that errors out (e.g. OOM) is recorded and skipped, not
fatal. Dropout is a fixed method constant (0.5) and is never searched.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import numpy as np
import torch

from ruleofthumb.embed import embed_texts
from ruleofthumb.explain import (
    Explainer,
    _is_path_batch,
    _is_string_batch,
    _path_loader,
    fit_image,
    fit_tabular,
    fit_text,
)

# Per-modality so the search can actually reach each one's working regime: text needs
# much smaller learning rates and batches (and l1 off), tabular needs the low rates and
# long fits the papers used. Ranges, not exact paper lookups.
DEFAULT_SPACES = {
    "tabular": {
        "learning_rate": [0.005, 0.01, 0.05, 0.1],
        "batch_size": [64, 500, 2000],
        "epochs": [100, 500, 1500],
        "weight_decay": [0.0, 0.01, 0.05],
    },
    "text": {
        "learning_rate": [1e-6, 1e-5, 1e-4, 0.01],
        "batch_size": [16, 32, 64],
        "epochs": [35, 100, 300],
        "weight_decay": [0.0, 0.01],
        "l1_penalty": [0.0, 0.01],
    },
    "image": {
        "learning_rate": [0.003, 0.01, 0.05],
        "batch_size": [500, 2000, 5000],
        "epochs": [100, 300, 600],
        "weight_decay": [0.0, 0.01, 0.05],
    },
}

# the undecorated factories: candidates never tune again (recursion guard for ``tune=True``)
_FACTORIES = {"tabular": fit_tabular, "text": fit_text, "image": fit_image}  # called with tune=None: the recursion guard
_LOADER_KWARGS = ("tokenizer", "model", "backbone", "size", "transform")


@dataclasses.dataclass(frozen=True)
class AutotuneResult:
    """Outcome of an :func:`autotune` search.

    Attributes:
        explainer: the winning configuration refit on **all** data.
        best_params: hyperparameters of the best validation candidate.
        best_score: held-out score of that candidate under the requested
            ``scoring`` (plain agreement by default, final-step reveal
            accuracy only when ``scoring="reveal"`` was named).
        trials: every candidate as ``{"params": ..., "score": ...}``,
            sorted best-first.
    """

    explainer: Explainer
    best_params: dict
    best_score: float
    trials: list


def _resolve_modality(x_inputs, modality):
    if modality != "auto":
        return modality
    if _is_path_batch(x_inputs):
        return "image"
    if _is_string_batch(x_inputs):
        return "text"
    ndim = np.asarray(x_inputs).ndim
    detected = {2: "tabular", 3: "text", 4: "image"}.get(ndim)
    if detected is None:
        raise ValueError(f"cannot infer a modality from input ndim={ndim}; pass modality= explicitly")
    return detected


def _split(n, validation_split, seed):
    rng = np.random.RandomState(seed)
    permuted = rng.permutation(n)
    n_val = max(1, round(n * validation_split))
    return permuted[n_val:], permuted[:n_val]


def _candidates(space, search, n_candidates, seed):
    keys = sorted(space)
    if search == "grid":
        return [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]
    rng = np.random.RandomState(seed)
    seen, combos, attempts = set(), [], 0
    while len(combos) < n_candidates and attempts < n_candidates * 20:
        attempts += 1
        combo = {key: space[key][int(rng.randint(len(space[key])))] for key in keys}
        marker = tuple(sorted(combo.items()))
        if marker not in seen:
            seen.add(marker)
            combos.append(combo)
    return combos


def _subset(inputs, indices):
    if isinstance(inputs, list):
        return [inputs[i] for i in indices]
    return inputs[indices]


def _is_oom(exc):
    """True for an out-of-memory error (CUDA or CPU); the only candidate failure autotune skips."""
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and any(s in message for s in ("out of memory", "allocate"))


def _split_value(value, indices, n):
    """Split a sample-aligned ``autotune`` kwarg alongside ``x`` / ``y``; pass the rest through."""
    if value is None or isinstance(value, (str, dict)):
        return value
    try:
        length = len(value)
    except TypeError:
        return value
    if length != n:
        return value
    return _subset(list(value) if isinstance(value, list) else np.asarray(value), indices)


def _split_kwargs(kwargs, indices, n):
    return {key: _split_value(value, indices, n) for key, value in kwargs.items()}


def _embed_once(modality, x_inputs, model_kwargs, device):
    """Embed raw strings / image paths once so candidates fit arrays instead of re-embedding each time."""
    if modality == "text":
        embedded = embed_texts(
            list(x_inputs), tokenizer=model_kwargs.get("tokenizer"), model=model_kwargs.get("model"), device=device
        )
        arrays, mask = embedded.embeddings, embedded.attention_mask
    else:
        loader, _ = _path_loader(
            model_kwargs.get("backbone", "auto"), model_kwargs.get("size"), model_kwargs.get("transform")
        )
        loaded = loader(list(x_inputs))
        arrays, mask = (loaded.maps if hasattr(loaded, "maps") else loaded.images), loaded.mask
    kwargs = {key: value for key, value in model_kwargs.items() if key not in _LOADER_KWARGS}
    return arrays, {**kwargs, "mask": mask}


def _agreement_score(explainer, x_val, y_val, mask=None):
    """Plain held-out agreement: surrogate predictions vs black-box labels.

    No reveal machinery runs here — this is the default candidate
    scoring, so tuning stays free of interventions unless asked.
    """
    kwargs = {"mask": mask} if mask is not None else {}
    preds = np.asarray(explainer.predict(x_val, **kwargs).cpu())
    return float((preds == np.asarray(y_val).flatten()).mean())


def _reveal_score(explainer, x_val, y_val, mask=None):
    """Final-step reveal accuracy on held-out data.

    Runs the reveal pipeline (`get_order` + `score_ordering`) against
    the surrogate — used only when ``scoring="reveal"`` is named.
    """
    order = explainer.get_order(x_val, mask=mask)
    curve = explainer.score_ordering(x_val, torch.from_numpy(np.asarray(y_val).astype(np.int64)), order)
    return float(curve[-1])


_SCORINGS = {"agreement": _agreement_score, "reveal": _reveal_score}


def autotune(
    y_outputs: Any,
    x_inputs: Any,
    *,
    modality: str = "auto",
    search: str = "random",
    n_candidates: int = 8,
    space: dict[str, list[Any]] | None = None,
    validation_split: float = 0.25,
    seed: int | None = None,
    device: Any | None = None,
    n_classes: int | None = None,
    scoring: str = "agreement",
    **model_kwargs: Any,
) -> AutotuneResult:
    """Search the training hyperparameters and return the best fitted explainer.

    Candidates are fitted on a seeded train split via the regular
    :func:`fit_tabular` / :func:`fit_text` / :func:`fit_image` factories,
    scored on the held-out split, and the winner is
    refit on **all** data. Each candidate gets its own seed (``seed + i``)
    so comparisons are fair.

    Args:
        y_outputs: black-box outputs; same semantics as the factories.
        x_inputs: arrays or raw strings / file paths (native ingestion is
            supported; padding masks are derived automatically). Strings /
            paths are embedded once for all candidates; the final refit
            embeds them again so the explainer accepts them back.
        modality: ``"auto"`` detects from the input, like :func:`ruleofthumb.fit`.
        search: ``"random"`` samples ``n_candidates`` unique combinations;
            ``"grid"`` enumerates every combination in ``space``.
        n_candidates: number of random-search candidates.
        space: hyperparameter search space; defaults to
            the modality's entry in :data:`DEFAULT_SPACES`. Keys must be a subset of it.
        validation_split: fraction of samples held out for scoring.
        scoring: ``"agreement"`` (default) scores candidates by plain
            held-out surrogate-vs-label agreement, running no reveal
            machinery; ``"reveal"`` scores by final-step reveal accuracy
            (the surrogate's predicted-class accuracy at full reveal)
            instead. Reveal-based scoring runs only when named here.
        seed: controls the split, candidate sampling and candidate fits.
        device: forwarded to the factories.
        n_classes: forwarded to the factories; inferred from
            ``len(unique(y_outputs))`` when omitted.
        **model_kwargs: forwarded to the factories for both the candidate
            fits and the final refit (e.g. ``nonlinear``, ``l1_penalty``,
            ``mask``, and ``patience`` / ``min_delta`` / ``check_every`` to
            plateau-stop every candidate and the refit — fit-to-convergence,
            ToDo 29). Sample-aligned array values (a
            ``mask`` with one entry per sample) are split alongside ``x`` /
            ``y``; scalars, strings and dicts pass through whole.

    Returns:
        :class:`AutotuneResult` with the full-data refit in ``.explainer``.
    """
    if search not in ("random", "grid"):
        raise ValueError(f"unknown search strategy: {search!r}")
    if scoring not in _SCORINGS:
        raise ValueError(f"unknown scoring: {scoring!r}; expected one of {sorted(_SCORINGS)}")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be in (0, 1)")

    modality = _resolve_modality(x_inputs, modality)
    factory = _FACTORIES[modality]
    default = DEFAULT_SPACES[modality]
    space = default if space is None else space
    unknown = set(space) - set(default)
    if unknown:
        raise ValueError(f"unknown hyperparameters in space for {modality}: {sorted(unknown)}")
    if _is_string_batch(x_inputs):
        inputs, candidate_kwargs = _embed_once(modality, x_inputs, model_kwargs, device)
    else:
        inputs, candidate_kwargs = np.asarray(x_inputs), model_kwargs
    labels = np.asarray(y_outputs).flatten()
    if n_classes is None:
        n_classes = len(np.unique(labels))

    train_idx, val_idx = _split(len(inputs), validation_split, seed)
    x_train, y_train = _subset(inputs, train_idx), labels[train_idx]
    x_val, y_val = _subset(inputs, val_idx), labels[val_idx]
    train_kwargs = _split_kwargs(candidate_kwargs, train_idx, len(inputs))
    val_mask = _split_value(candidate_kwargs.get("mask"), val_idx, len(inputs))

    def evaluate(params, seed_i, epochs=None):
        """Fit one candidate (optionally at a reduced epoch budget) and score it; OOM is skipped, not fatal."""
        call = {**train_kwargs, **params}  # searched params win over caller-pinned kwargs; screen epochs win over both
        if epochs is not None:
            call["epochs"] = epochs
        try:  # guard the fit only (where OOM happens); a scoring error is a real bug and propagates
            candidate = factory(y_train, x_train, tune=None, seed=seed_i, device=device, n_classes=n_classes, **call)
        except (RuntimeError, MemoryError) as exc:  # an OOM candidate must not kill the search; real bugs still raise
            if not _is_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"params": params, "score": float("-inf"), "error": repr(exc)}
        # ponytail: an OOM during scoring (val predict) still propagates; fit is the expensive path
        return {"params": params, "score": _SCORINGS[scoring](candidate, x_val, y_val, mask=val_mask)}

    candidates = _candidates(space, search, n_candidates, seed)
    seeds = [None if seed is None else seed + i for i in range(len(candidates))]
    if len(candidates) >= 4:
        # Successive halving (ToDo 29): screen every candidate at ~1/3 of its epochs, then finish only the
        # better-scoring half at full epochs. Abandoned candidates keep their cheap screen score.
        # ponytail: fixed 1/3 rung and top-half keep; a multi-rung schedule only pays off past ~tens of candidates.
        screen = [evaluate(p, s, epochs=max(1, p.get("epochs", 300) // 3)) for p, s in zip(candidates, seeds)]
        ranked = sorted(range(len(candidates)), key=lambda j: screen[j]["score"], reverse=True)
        survivors = set(ranked[: (len(candidates) + 1) // 2])
        trials = [evaluate(candidates[j], seeds[j]) if j in survivors else {**screen[j], "pruned": True}
                  for j in range(len(candidates))]
    else:
        trials = [evaluate(p, s) for p, s in zip(candidates, seeds)]
    if all(trial["score"] == float("-inf") for trial in trials):
        raise RuntimeError("every autotune candidate failed; see trial['error'] in the result's .trials")
    # best score first; ties break toward the cheaper fit (fewer epochs, then larger batch) for determinism
    trials.sort(key=lambda trial: (-trial["score"], trial["params"].get("epochs", 0), -trial["params"].get("batch_size", 0)))

    best = trials[0]
    explainer = factory(y_outputs, x_inputs, tune=None, seed=seed, device=device, n_classes=n_classes, **model_kwargs, **best["params"])
    return AutotuneResult(explainer=explainer, best_params=best["params"], best_score=best["score"], trials=trials)
