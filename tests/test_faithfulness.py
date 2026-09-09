"""Faithfulness probes on synthetic fixtures with known mechanisms.

Deletion (mask top-k by |importance|, agreement must drop), pointing-game
(argmax inside the known signal box) and noise suppression (noise columns
rank below signal) — distilled from the sandbox legit/DS tracks to fast,
deterministic, offline unit tests.
"""

import numpy as np
import torch

from ruleofthumb import fit_image, fit_tabular


def test_deletion_gap_tabular():
    rng = np.random.RandomState(2)
    x = rng.rand(128, 4).astype(np.float32)
    y = ((x[:, 0] + x[:, 1]) > 1.0).astype(np.int64)
    exp = fit_tabular(y, x, epochs=60, batch_size=64, learning_rate=0.05, seed=0)

    full = float((np.asarray(exp.predict(torch.from_numpy(x)).cpu()) == y).mean())
    top2 = np.argsort(-np.abs(exp.get_explanation(x)).mean(0))[:2]
    assert set(top2.tolist()) == {0, 1}
    knocked = x.copy()
    knocked[:, top2] = 0.0
    dropped = float((np.asarray(exp.predict(torch.from_numpy(knocked)).cpu()) == y).mean())
    assert full - dropped >= 0.25


def test_pointing_game_localized_square():
    rng = np.random.RandomState(3)
    n = 8
    x = rng.randn(n, 1, 16, 16).astype(np.float32) * 0.2
    y = np.array([i % 2 for i in range(n)], dtype=np.int64)
    x[y == 1, 0, 4:8, 4:8] += 3.0  # bright square = the class signal
    exp = fit_image(y, x, epochs=60, batch_size=8, learning_rate=0.05, seed=0)

    imp = exp.get_explanation(x)
    hits = 0
    for i in np.flatnonzero(y == 1):
        r, c = np.unravel_index(int(np.argmax(np.abs(imp[i]))), (16, 16))
        hits += 4 <= r < 8 and 4 <= c < 8
    assert hits / max(1, int((y == 1).sum())) >= 0.75


def test_noise_columns_rank_below_signal():
    rng = np.random.RandomState(4)
    x = rng.randn(64, 6).astype(np.float32)
    y = ((2 * x[:, 0] - 1.5 * x[:, 1]) > 0).astype(np.int64)
    exp = fit_tabular(y, x, epochs=100, batch_size=64, learning_rate=0.05, seed=0)

    scores = np.abs(exp.get_explanation(x)).mean(0)
    assert float(scores[:2].min()) > float(scores[2:].max())
