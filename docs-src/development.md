# Development guide

How this repo is built, checked, and released. Start here before your
first commit; the release ritual (§Releasing) is the only part with
sharp edges.

## Setup

Requires Python 3.10 and `uv`:

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e ".[dev,docs]"
```

`requirements.txt` pins the exact tested environment; `pyproject.toml`
carries only lower bounds. All commands below run from the repo root
with `.venv` active (or prefixed `.venv/bin/`).

## Everyday checks

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check .
mkdir -p docs-src/notebooks && cp examples/0*.ipynb docs-src/notebooks/ && .venv/bin/sphinx-build -W docs-src /tmp/rot-site
```

Green on all three is the definition of done, enforced mechanically by
CI on every push and pull request. The test suite never trains a model
and never downloads a dataset — it loads committed artifacts plus
locally cached weights only (backbone revisions are pinned; see
`tests/integration/conftest.py`). Integration fits run on CPU by default
for exact cross-machine reproduction; GPU coverage is split out:

- `tests/integration/test_device_parity.py` cross-checks each locally
  available accelerator (MPS/CUDA) against CPU within tolerance — a plain
  local `pytest` exercises MPS automatically on Apple Silicon, while
  CPU-only CI runs the CPU leg and skips the rest.
- To run the whole integration tier on an accelerator instead
  (exploratory): `ROT_TEST_DEVICE=mps pytest tests/integration -q
  -p no:cacheprovider` (or `=cuda` where available). Docs are written in
  MyST markdown under `docs-src/`; cross-page anchors need explicit
  `(label)=` targets (plain `page.md#header` links fail `-W`).

The pets heatmap anchor loads committed fitted weights
(`pet_rot_state.pt`, raw state dict — never `.rotx`, whose version stamp
would break the fixture on every release) instead of refitting: the
300-epoch fit amplifies last-ulp host differences across machines, while
the forward pass is stable to ulp level. Remint both artifacts together
with `tests/integration/mint_pet_weights.py` if the model or fit changes.

## Automation map

| Event | What runs | Effect |
|---|---|---|
| Push / pull request | `check.yml`: ruff + pytest + docs build | Red status blocks the merge; nothing deploys |
| Push to `main` | `testpypi.yml`: stamp `<base>.dev<RUN_NUMBER>`, build + upload | Per-commit TestPyPI release; packaging breakage surfaces immediately |
| Push of tag `v*` | `release.yml`: build + upload to PyPI | Live release; the tag is the only trigger |
| Push to `main` / new tag | ReadTheDocs | `latest` rebuilds on push; `stable` follows activated tags |

Nothing publishes to live PyPI on a branch push, a PR merge, or a
version bump — only a tag. Commit-message flags do not exist and must
not be invented: if it isn't tagged, it isn't released.

## Releasing

1. Make the change; update README (usage + migration notes).
2. Bump all three version sites: `pyproject.toml`,
   `src/ruleofthumb/__init__.py`, the assertion in
   `tests/test_explain.py`.
3. Add a `ToDo.md` changelog entry newest-first, directly under
   `## Changelog`.
4. Run the §Everyday checks trio locally; rebuild docs to `/tmp`.
5. Commit, review `git status`, push to `main`. CI runs checks, uploads
   to TestPyPI, and RTD rebuilds `latest` — confirm each is green.
6. Tag and push the tag: `git tag vX.Y.Z && git push origin main --tags`.
   The release workflow uploads to PyPI.
7. Confirm on `https://pypi.org/project/ruleofthumb-rot/`: version,
   MIT licence, README banner, all four `project.urls`.
8. Paste the `ToDo.md` changelog entry into the GitHub release notes.
9. In RTD (`Admin → Versions`), activate the new tag so `stable`
   tracks the release.

Never reuse a published version number: if the wheel is broken, yank it
and ship the fix as the next version.

## Trusted publishing (one-time setup)

Uploads use OpenID Connect — there are no API tokens to mint, store, or
rotate. Each index needs its publisher registered once (only possible
after the project exists there, so the manual first upload came first):

- **PyPI**: project page → Settings → Publishing → Add a new publisher:
  owner `KaiRawal`, repository `Rule-of-Thumb`, workflow
  `release.yml`. No environment.
- **TestPyPI**: same steps on `test.pypi.org` with workflow
  `testpypi.yml`.

The workflows request `id-token: write` and nothing else. If a publish
job 403s, the publisher registration (names must match exactly) is the
first suspect, not credentials — there are none.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check.yml` red on pytest/ruff/docs | Reproduce with the §Everyday checks trio; fix, don't re-run blindly. |
| `testpypi.yml` / `release.yml` 403 | Publisher misconfigured (owner/repo/workflow typo) or version number reused — PyPI never reuses versions. |
| Release workflow never fires | Tags must match `v*` and be pushed (`git push --tags`); a local-only tag does nothing. |
| RTD build red | Reproduce with the §1 docs command; RTD builds `requirements.txt` on Ubuntu — anything machine-specific fails there first. |
| Stale notebook output on RTD | RTD executes notebooks itself via the `pre_build` staging job; clear local `.jupyter_cache/` and rebuild if yours disagree. |
