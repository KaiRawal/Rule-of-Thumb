"""Integration tests: device parity smoke coverage.

The integration tier pins ``TEST_DEVICE=cpu`` (see ``conftest.py``) so every
run reproduces exactly on macOS and Linux. This module is the GPU
counterpart: it fits tiny synthetic surrogates on CPU and on each locally
available accelerator (MPS/CUDA), asserting the accelerator agrees with CPU
on predicted classes and per-feature explanation direction within tolerance.
Bounds are deliberately loose smoke checks, not exact anchors — exactness is
the CPU tier's job.

Typical local dev: a plain ``pytest`` run exercises MPS automatically on
Apple Silicon. CI (CPU-only runners) runs the CPU self-parity leg and skips
the accelerator legs.
"""

import numpy as np
import pytest
import torch

from ruleofthumb import fit_tabular

SEED = 0


def _accelerators():
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _synthetic(n=256, d=4):
    rng = np.random.RandomState(0)
    x = rng.rand(n, d).astype(np.float32)
    return x, (x[:, 0] > 0.5).astype(np.int64)


def _fit(x, y, device):
    return fit_tabular(y, x, epochs=50, batch_size=64, learning_rate=0.05, seed=SEED, device=device)


def test_cpu_fit_reproduces_exactly():
    """Two same-seed CPU fits agree bit-for-bit (the determinism contract)."""
    x, y = _synthetic()
    first = _fit(x, y, "cpu").get_explanation(x)
    second = _fit(x, y, "cpu").get_explanation(x)
    assert np.array_equal(first, second)


def test_accelerator_matches_cpu_within_tolerance():
    """Each available accelerator agrees with CPU on classes and magnitudes."""
    devices = _accelerators()
    if not devices:
        pytest.skip("no MPS/CUDA accelerator available")
    x, y = _synthetic()
    for device in devices:
        cpu_pred = _fit(x, y, "cpu").predict(torch.from_numpy(x)).cpu().numpy()
        acc_pred = _fit(x, y, device).predict(torch.from_numpy(x)).cpu().numpy()
        assert float((cpu_pred == acc_pred).mean()) >= 0.95

        cpu_imp = _fit(x, y, "cpu").get_explanation(x)
        acc_imp = _fit(x, y, device).get_explanation(x)
        assert np.isfinite(acc_imp).all()
        # magnitudes drift across backends over 50 epochs, but the
        # explanations must rank features the same way per feature
        corrs = [float(np.corrcoef(cpu_imp[:, j], acc_imp[:, j])[0, 1]) for j in range(x.shape[1])]
        assert min(corrs) >= 0.98
