import os

import pytest

from ruleofthumb import core, embed

TEST_DEVICE = os.environ.get("ROT_TEST_DEVICE", "cpu")


@pytest.fixture(scope="session", autouse=True)
def _pin_default_device():
    """Run fits that omit ``device=`` on ``ROT_TEST_DEVICE`` (default CPU), like the integration tier.

    Without this, a GPU host auto-selects CUDA, whose RNG streams differ from
    CPU's, so live-fit floors measured on CPU fail there. Explicit
    ``device=`` arguments (e.g. the accelerator parity test) are untouched.
    """
    resolve = core._resolve_device

    def pinned(device=None):
        return resolve(TEST_DEVICE if device is None else device)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "_resolve_device", pinned)
        mp.setattr(embed, "_resolve_device", pinned)
        yield
