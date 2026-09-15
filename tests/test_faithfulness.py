"""Faithfulness probes on synthetic fixtures with known mechanisms.

Deletion (mask top-k by |importance|, agreement must drop), pointing-game
(argmax inside the known signal box) and noise suppression (noise columns
rank below signal) — distilled from the sandbox legit/DS tracks to fast,
deterministic, offline unit tests.

Fits are committed mock weights (``tests/_mock_weights.py``): only the
deterministic forward pass runs at test time, so the margins below are
tight. Dataset builders are shared with the mint script by import.
"""

import numpy as np
import torch
from _mock_weights import data_faith_deletion, data_faith_noise, data_faith_pointing, load_mock


def test_deletion_gap_tabular():
    x, _, y = data_faith_deletion()
    exp = load_mock("faith_deletion")

    full = float((np.asarray(exp.predict(torch.from_numpy(x)).cpu()) == y).mean())
    top2 = np.argsort(-np.abs(exp.get_explanation(x)).mean(0))[:2]
    assert set(top2.tolist()) == {0, 1}
    knocked = x.copy()
    knocked[:, top2] = 0.0
    dropped = float((np.asarray(exp.predict(torch.from_numpy(knocked)).cpu()) == y).mean())
    assert full - dropped >= 0.35


def test_pointing_game_localized_square():
    x, _, y = data_faith_pointing()
    exp = load_mock("faith_pointing")

    imp = exp.get_explanation(x)
    hits = 0
    for i in np.flatnonzero(y == 1):
        r, c = np.unravel_index(int(np.argmax(np.abs(imp[i]))), (16, 16))
        hits += 4 <= r < 8 and 4 <= c < 8
    assert hits / max(1, int((y == 1).sum())) >= 0.9


def test_noise_columns_rank_below_signal():
    x, _, _ = data_faith_noise()
    exp = load_mock("faith_noise")

    scores = np.abs(exp.get_explanation(x)).mean(0)
    assert float(scores[:2].min()) > float(scores[2:].max())
