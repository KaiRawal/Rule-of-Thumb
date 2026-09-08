# Publishing `ruleofthumb` v0.0.1 to PyPI (+ docs to Pages)

> ⚠️ Experimental 0.0.x pre-alpha — entirely vibe-coded from hand-written
> research code. May break; backwards-incompatible changes expected.

This runbook takes the repo from "ready" to "public". Nothing here
publishes automatically — every upload/tag/push is a manual command you
run. Do not commit from automation; review `git status` first.

## 0. Prerequisites

- PyPI account + API token (`Account Settings → API tokens`). Store as
  `$PYPI_TOKEN`; for the dry run, a TestPyPI token as `$TEST_PYPI_TOKEN`.
- `uv` installed (this repo uses the `uv_build` backend; `uv build` is
  the only supported build command).
- Clean tree on `main`.
- Version pinned in all three places: `pyproject.toml`,
  `src/ruleofthumb/__init__.py`, `tests/test_explain.py` (all `0.0.1`).

## 1. Final verification (repo root, venv only)

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
uv build
tar tzf dist/ruleofthumb-0.0.1.tar.gz | head -n 20  # LICENSE + README present
pip install dist/ruleofthumb-0.0.1-py3-none-any.whl --force-reinstall
python -c "import ruleofthumb; print(ruleofthumb.__version__)"
```

Expected: pytest green, ruff clean, `dist/` holds exactly one sdist +
one wheel, import prints `0.0.1`. Delete and rebuild `dist/` if stale
artifacts linger (`rm -rf dist/ && uv build`).

Docs build (one command, venv only — stages the `examples/` notebooks as
generated copies under `docs-src/notebooks/`, re-executes them, validates
links, writes the committed site to `docs/`):

```bash
.venv/bin/pip install -e ".[docs]" && mkdir -p docs-src/notebooks && cp examples/0*.ipynb docs-src/notebooks/ && .venv/bin/mkdocs build --strict && touch docs/.nojekyll
```

## 2. Dry run: TestPyPI

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN"
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ruleofthumb==0.0.1
python -c "import ruleofthumb; print(ruleofthumb.__version__)"
```

Check the TestPyPI project page renders: title, README (with the
experimental banner), MIT licence, Homepage/Repository/Issues/
Documentation links.

## 3. Publish: PyPI

```bash
uv publish --token "$PYPI_TOKEN"
```

Then confirm on `https://pypi.org/p/ruleofthumb/`:

- Version `0.0.1`, licence MIT, Python `>=3.9`.
- README banner visible; `project.urls` all resolve.

## 4. Tag and push (manual)

```bash
git status --short   # review before committing
git add -A
git commit -m "feat: first public pre-alpha v0.0.1 (uv_build, docs, PyPI metadata)"
git tag v0.0.1
git push origin main --tags
```

Open the release notes from the tag, pasting the `ToDo.md`
`v0.0.1` changelog entry.

## 5. Docs to GitHub Pages (manual, one time setup + per release)

No CI workflows in this repo (deferred with ToDo item 17) — deploy by
hand from the repo root. The built site in `docs/` is committed; Pages
serves this branch's `/docs` folder.

```bash
# 1. One-command site build (notebooks staged + executed, links validated):
.venv/bin/pip install -e ".[docs]" && mkdir -p docs-src/notebooks && cp examples/0*.ipynb docs-src/notebooks/ && .venv/bin/mkdocs build --strict && touch docs/.nojekyll
# 2. Sanity-check the output locally:
.venv/bin/mkdocs serve   # browse http://127.0.0.1:8000 — Home, Guides,
                         # executed Notebooks, Migration, Capacity, API, Test report
# 3. Commit the rebuilt site and push:
git add docs docs-src mkdocs.yml
git commit -m "docs: rebuild site for v0.0.1"
git push origin main
```

Then enable Pages (`Settings → Pages → Deploy from a branch → main` →
`/docs`), and confirm the served URL matches `[project.urls]
Documentation` in `pyproject.toml`.

Notes:

- Expected build noise (safe to ignore): `[IPKernelApp] WARNING | Kernel
  is running over TCP...` lines from notebook execution, and the red
  "Material for MkDocs" banner about MkDocs 2.0 (upstream notice, not a
  build warning). `mkdocs build --strict` still fails on any real warning
  — a green build ends with `Documentation built`.
- `docs-src/notebooks/` are gitignored generated copies — never commit
  them. The canonical notebooks live in `examples/`; the built HTML in
  `docs/` (plus `docs/.nojekyll`) is committed.
- Re-run the one-command build and commit `docs/` before each release so
  the executed notebooks and the included `tests/TEST_SUITE.md` report
  stay fresh.

## 6. After the release

- Yank/re-publish only if the wheel is broken (`yank`, never reuse
  `0.0.1` — bump to `0.0.2`).
- Next version: bump all three version sites, add a `ToDo.md` changelog
  entry newest-first, rebuild + commit docs.

## 7. If something fails

| Symptom | Fix |
|---|---|
| `uv build` errors on backend | Check `[build-system]` = `uv_build>=0.12.10,<0.13` / `uv_build`; run with `RUST_LOG=uv=debug uv build`. |
| Sdist missing `LICENSE`/`README` | Check `license-files = ["LICENSE*"]` and `readme = "README.md"`; `LICENSE` must sit next to `pyproject.toml`. |
| `uv publish` 403 | Token scope (project vs account), or name taken — confirm `pypi.org/p/ruleofthumb/`. |
| `mkdocs build --strict` warnings as errors | Fix the flagged link/docstring, don't drop `--strict`. |
| Pages serves 404 / raw files | `docs/.nojekyll` must be committed; Pages source must be `main` → `/docs`. |
| Heavy install (`torch`, `transformers`, `shap`) surprises users | Known single-install decision (AGENTS.md); point to README install notes. |
