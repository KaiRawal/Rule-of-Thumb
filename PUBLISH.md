# Publishing `ruleofthumb-rot` to PyPI (+ docs on ReadTheDocs)

> ⚠️ Experimental 0.0.x pre-alpha — entirely vibe-coded from hand-written
> research code. May break; backwards-incompatible changes expected.

This runbook takes the repo from "ready" to "public". TestPyPI and PyPI
uploads run in CI (see the development guide); tagging, pushing, and the
dashboard steps stay manual. Do not commit from automation; review
`git status` first.

## 0. Prerequisites

- Trusted publishers registered (one-time setup, see the development
  guide): PyPI → `release.yml`, TestPyPI → `testpypi.yml`. No API tokens
  are used anywhere.
- `uv` installed (this repo uses the `uv_build` backend; `uv build` is
  the only supported build command).
- ReadTheDocs account with this repo connected (one-time setup, see §5).
- Clean tree on `main`.
- Version pinned in all three places: `pyproject.toml`,
  `src/ruleofthumb/__init__.py`, `tests/test_explain.py` (all the same
  `X.Y.Z`).

## 1. Final verification (repo root, venv only)

```bash
VER="$(grep -m1 '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')"
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
uv build
tar tzf dist/ruleofthumb_rot-$VER.tar.gz | head -n 20  # LICENSE + README present
pip install dist/ruleofthumb_rot-$VER-py3-none-any.whl --force-reinstall
python -c "import ruleofthumb as rot; print(rot.__version__)"
```

Expected: pytest green, ruff clean, `dist/` holds exactly one sdist +
one wheel, import prints `$VER`. Delete and rebuild `dist/` if stale
artifacts linger (`rm -rf dist/ && uv build`).

Docs build (one command, venv only — stages the `examples/` notebooks as
generated copies under `docs-src/notebooks/`, re-executes them with the
execution cache, validates links; output goes to a throwaway dir, nothing
is committed):

```bash
.venv/bin/pip install -e ".[docs]" && mkdir -p docs-src/notebooks && cp examples/0*.ipynb docs-src/notebooks/ && .venv/bin/sphinx-build -W docs-src /tmp/rot-site
```

## 2. Dry run: TestPyPI (automatic on every `main` push)

`testpypi.yml` stamps a per-commit dev version (`<base>.dev<RUN_NUMBER>`,
patched into `pyproject.toml` + `__version__` in the CI workspace only —
the repo keeps its release version) so every merge uploads a unique
distribution; re-runs fall back to `skip-existing`. Check the run, then the
project page. Manual fallback (same commands CI runs):

```bash
uv build
uv publish --publish-url https://test.pypi.org/legacy/
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ruleofthumb-rot==<version>
python -c "import ruleofthumb as rot; print(rot.__version__)"
```

Check the TestPyPI project page renders: title, README (with the
experimental banner), MIT licence, Homepage/Repository/Issues/
Documentation links.

## 3. Publish: PyPI (automatic on `v*` tags)

Pushing the tag from §4 triggers `release.yml`, which builds and uploads
via trusted publishing. Manual fallback: `uv build` then `uv publish`.

Then confirm on `https://pypi.org/p/ruleofthumb-rot/`:

- Version `X.Y.Z`, licence MIT, Python `>=3.9`.
- README banner visible; `project.urls` all resolve.

## 4. Tag and push (manual — the release trigger)

```bash
git status --short   # review before committing
git add -A
git commit -m "release X.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

Open the release notes from the tag, pasting the matching `ToDo.md`
changelog entry.

## 5. Docs on ReadTheDocs (one-time setup, then automatic)

No built HTML is ever committed — RTD builds the site itself from
`docs-src/` + `.readthedocs.yaml` on every push and tag. GitHub Actions
handles checks and package uploads (see the development guide); RTD
handles docs.

One-time setup (in the RTD dashboard):

1. `Import a Project` → connect `KaiRawal/Rule-of-Thumb`.
2. Confirm the build config is picked up (`.readthedocs.yaml`: Python
   3.10, `requirements.txt`, `pip install .`, Sphinx at
   `docs-src/conf.py`).
3. First build runs automatically; confirm the site renders at
   `https://rule-of-thumb.readthedocs.io/` — Home, Guides, executed
   Notebooks, Migration, Capacity, API, Test report.
4. Under `Admin → Versions`, activate the new tag build so
   `stable` tracks the release; `latest` tracks `main`.

Per release, nothing docs-specific is required: pushing the tag rebuilds
`stable` automatically. To preview doc changes before release, open a PR —
RTD builds PR previews.

Local preview (same one-command build as §1, then serve the output):

```bash
.venv/bin/pip install -e ".[docs]" && mkdir -p docs-src/notebooks && cp examples/0*.ipynb docs-src/notebooks/ && .venv/bin/sphinx-build -W docs-src /tmp/rot-site && .venv/bin/python -m http.server -d /tmp/rot-site 8000
```

Notes:

- `sphinx-build -W` turns warnings into errors — fix the flagged
  link/docstring, never drop `-W`. A separate `sphinx-build -b linkcheck`
  run catches external-link rot.
- `docs-src/notebooks/` are gitignored generated copies — never commit
  them. The canonical notebooks live in `examples/`; notebook outputs are
  cached in `.jupyter_cache/` (also gitignored) so rebuilds only re-run
  changed notebooks.
- Expected build noise (safe to ignore): `[IPKernelApp] WARNING | Kernel
  is running over TCP...` lines from notebook execution. A green build
  ends with `build succeeded`.

## 6. Managing upgrades

### New package version (e.g. `X.Y.Z`)

1. Make the behaviour change; update README (usage + migration notes).
2. Bump all three version sites: `pyproject.toml`,
   `src/ruleofthumb/__init__.py`, the assertion in
   `tests/test_explain.py`.
3. Add a `ToDo.md` changelog entry newest-first (directly under
   `## Changelog`), describing the change and any legacy departure.
4. Re-run §1 (pytest, ruff, `uv build`, docs build to `/tmp/rot-site`).
5. Publish via §§2–4; RTD rebuilds `stable` from the new tag on its own.

### Dependency upgrades

`requirements.txt` pins the exact tested environment; `pyproject.toml`
carries only lower bounds. To upgrade a dependency:

```bash
python -m venv /tmp/rot-upgrade --clear   # scratch env, never the repo .venv
/tmp/rot-upgrade/bin/pip install -r requirements.txt
/tmp/rot-upgrade/bin/pip install -U <package>          # or -U --all-targets? no: one at a time
.venv/bin/python -m pytest   # in the repo venv after applying: see below
```

Concretely: bump the pin in `requirements.txt`, reinstall the repo venv
(`uv pip install --python .venv/bin/python -r requirements.txt`), run the
full §1 verification, and commit `requirements.txt` (+ `pyproject.toml`
bounds if the upgrade needs a new minimum). One dependency per commit so
bisects stay useful. Never reuse a published version number after a
dependency change — if the current version is already on PyPI, the upgrade
ships as the next version per §6 above.

Docs-stack upgrades (`sphinx`, `myst-nb`, `pydata-sphinx-theme`,
`sphinx-copybutton`) follow the same flow, verified by the §1 docs build
instead of pytest; then confirm the RTD `latest` build for the push.

## 7. If something fails

| Symptom | Fix |
|---|---|
| `uv build` errors on backend | Check `[build-system]` = `uv_build>=0.12.10,<0.13` / `uv_build`; run with `RUST_LOG=uv=debug uv build`. |
| Sdist missing `LICENSE`/`README` | Check `license-files = ["LICENSE*"]` and `readme = "README.md"`; `LICENSE` must sit next to `pyproject.toml`. |
| CI publish job 403 | Trusted publisher misconfigured (owner/repo/workflow typo) or version number reused — PyPI never reuses versions. There are no tokens to expire. |
| `sphinx-build -W` warnings as errors | Fix the flagged link/docstring, don't drop `-W`. |
| Stale notebook outputs in the site | Clear the execution cache (`rm -rf .jupyter_cache`) and rebuild; changed notebooks re-run automatically. |
| RTD build fails | Check `.readthedocs.yaml` (Python version, requirements path); reproduce locally with the §1 command; RTD needs `requirements.txt` to install cleanly on Ubuntu. |
| Heavy install (`torch`, `transformers`, `shap`) surprises users | Known single-install decision (AGENTS.md); point to README install notes. |
