"""Install a workflow-minted pet reference anchor.

Usage (repo root, venv only)::

    .venv/bin/python tests/integration/install_pet_reference.py <downloaded .npz>

Takes the ``pet_reference_explanations.x86_64.npz`` file downloaded from a
``mint-pet-reference`` workflow run, verifies its shape, copies it into
``tests/integration/artifacts/``, updates ``manifest.json``, and prints the
commit command. Only ``x86_64`` files are accepted (arm64 is minted locally
with ``mint_pet_reference.py``, never downloaded).
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
EXPECTED_NAME = "pet_reference_explanations.x86_64.npz"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    src = sys.argv[1]
    if os.path.basename(src) != EXPECTED_NAME:
        print(f"refusing {src!r}: expected exactly {EXPECTED_NAME}")
        raise SystemExit(1)
    data = np.load(src)
    heatmaps = data["heatmaps"]
    assert heatmaps.shape == (20, 7, 7), heatmaps.shape
    assert heatmaps.dtype == np.float32, heatmaps.dtype
    assert np.isfinite(heatmaps).all()

    dst = os.path.join(ARTIFACTS, EXPECTED_NAME)
    shutil.copyfile(src, dst)

    from generate_artifacts import record

    manifest_path = os.path.join(ARTIFACTS, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    record(manifest, EXPECTED_NAME, {"heatmaps": heatmaps})
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    sha = hashlib.sha256(Path(dst).read_bytes()).hexdigest()
    print(f"installed {dst}")
    print(f"sha256: {sha}")
    print("next: git status --short && git add -A && git commit -m ... && git push")


if __name__ == "__main__":
    main()
