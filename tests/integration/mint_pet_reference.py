"""Mint the platform-specific pet reference anchor.

Usage (repo root, venv only)::

    .venv/bin/python tests/integration/mint_pet_reference.py [--check]

Computes MobileNet features through ``_pet_features`` (the same code as the
test fixture), applies the same seeding/threading as the ``_deterministic``
fixture, runs the test's exact ``_fit_pets`` fit, and writes
``artifacts/pet_reference_explanations.<arch>.npz`` plus its sha256.

``--check`` mints to a temp file and verifies it matches the committed
anchor byte-for-byte, proving the script reproduces the test path (used in
CI review and after refactors; exits nonzero on mismatch).
"""

import hashlib
import os
import platform
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _arch():
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    raise RuntimeError(f"unsupported architecture {machine!r} for pet reference minting")


def _seed_everything():
    seed = int(os.environ.get("ROT_TEST_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def mint(path):
    from _pet_features import compute_pet_features
    from test_gpt_pet_integration import _fit_pets

    _seed_everything()
    labels = pd.read_csv(os.path.join(ARTIFACTS, "pets_labels.csv"))
    features = compute_pet_features(os.path.join(ARTIFACTS, "pet_images"), labels["filename"])
    y_gpt = labels["gpt_label"].eq("dog").to_numpy().astype(np.int64)
    heatmaps = _fit_pets(features, y_gpt).get_explanation(features).astype(np.float32)
    assert heatmaps.shape == (20, 7, 7)
    np.savez_compressed(path, heatmaps=heatmaps)
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    print(f"minted {path}")
    print(f"sha256: {sha}")
    return sha


def main():
    arch = _arch()
    target = os.path.join(ARTIFACTS, f"pet_reference_explanations.{arch}.npz")
    if "--check" in sys.argv[1:]:
        with tempfile.TemporaryDirectory() as tmp:
            probe = os.path.join(tmp, "probe.npz")
            mint(probe)
            committed = Path(target).read_bytes()
            fresh = Path(probe).read_bytes()
            if committed != fresh:
                print(f"MISMATCH: {target} differs from a fresh mint on this machine")
                raise SystemExit(1)
            print(f"OK: {target} reproduces bit-for-bit on {arch}")
    else:
        mint(target)


if __name__ == "__main__":
    main()
