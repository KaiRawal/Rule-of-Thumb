"""Automatic hyperparameter tuning for RoT explainers.

Keras-tuner-style search over the exposed training hyperparameters
(``learning_rate``, ``batch_size``, ``epochs``, ``dropout_rate``,
``weight_decay``): candidates are fitted on a seeded train split, scored on
held-out data by final-step reveal fidelity, and the winner is refit on all
data.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import numpy as np
import torch

from ruleofthumb.explain import Explainer, _is_path_batch, _is_string_batch, fit_image, fit_tabular, fit_text

DEFAULT_SPACE = {
    "learning_rate": [0.003, 0.01, 0.03, 0.1],
    "batch_size": [64, 500, 2000],
    "epochs": [100, 300, 600],
    "dropout_rate": [0.1, 0.3, 0.5],
    "weight_decay": [0.0, 0.01, 0.05],
}

_FACTORIES = {"tabular": fit_tabular, "text": fit_text, "image": fit_image}


@dataclasses.dataclass(frozen=True)
class AutotuneResult:
    """Outcome of an :func:`autotune` search.

    Attributes:
        explainer: the winning configuration refit on **all** data.
        best_params: hyperparameters of the best validation candidate.
        best_score: held-out final-step reveal accuracy of that candidate.
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


def _validation_score(explainer, x_val, y_val):
    """Final-step reveal accuracy on held-out data."""
    order = explainer.get_order(x_val)
    curve = explainer.score_ordering(x_val, torch.from_numpy(np.asarray(y_val).astype(np.int64)), order)
    return float(curve[-1])


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
) -> AutotuneResult:
    """Search the training hyperparameters and return the best fitted explainer.

    Candidates are fitted on a seeded train split via the regular
    :func:`fit_tabular` / :func:`fit_text` / :func:`fit_image` factories,
    scored on the held-out split by final-step reveal accuracy (the
    surrogate's predicted-class accuracy at full reveal), and the winner is
    refit on **all** data. Each candidate gets its own seed (``seed + i``)
    so comparisons are fair.

    Args:
        y_outputs: black-box outputs; same semantics as the factories.
        x_inputs: arrays or raw strings / file paths (native ingestion is
            supported; padding masks are derived automatically).
        modality: ``"auto"`` detects from the input, like :func:`ruleofthumb.fit`.
        search: ``"random"`` samples ``n_candidates`` unique combinations;
            ``"grid"`` enumerates every combination in ``space``.
        n_candidates: number of random-search candidates.
        space: hyperparameter search space; defaults to
            :data:`DEFAULT_SPACE`. Keys must be a subset of it.
        validation_split: fraction of samples held out for scoring.
        seed: controls the split, candidate sampling and candidate fits.
        device: forwarded to the factories.

    Returns:
        :class:`AutotuneResult` with the full-data refit in ``.explainer``.
    """
    if search not in ("random", "grid"):
        raise ValueError(f"unknown search strategy: {search!r}")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be in (0, 1)")
    space = DEFAULT_SPACE if space is None else space
    unknown = set(space) - set(DEFAULT_SPACE)
    if unknown:
        raise ValueError(f"unknown hyperparameters in space: {sorted(unknown)}")

    modality = _resolve_modality(x_inputs, modality)
    factory = _FACTORIES[modality]
    inputs = list(x_inputs) if _is_string_batch(x_inputs) else np.asarray(x_inputs)
    labels = np.asarray(y_outputs).flatten()

    train_idx, val_idx = _split(len(inputs), validation_split, seed)
    x_train, y_train = _subset(inputs, train_idx), labels[train_idx]
    x_val, y_val = _subset(inputs, val_idx), labels[val_idx]

    trials = []
    for i, params in enumerate(_candidates(space, search, n_candidates, seed)):
        candidate_seed = None if seed is None else seed + i
        candidate = factory(y_train, x_train, seed=candidate_seed, device=device, **params)
        score = _validation_score(candidate, x_val, y_val)
        trials.append({"params": params, "score": score})
    trials.sort(key=lambda trial: trial["score"], reverse=True)

    best = trials[0]
    explainer = factory(y_outputs, x_inputs, seed=seed, device=device, **best["params"])
    return AutotuneResult(explainer=explainer, best_params=best["params"], best_score=best["score"], trials=trials)
