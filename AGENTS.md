# AGENTS.md — ruleofthumb (standalone repo)

Instructions for AI coding agents working on the `ruleofthumb` pip package.
Everything in this file is binding: if your planned change conflicts with it,
stop and ask the maintainer.

## Project overview

This repo (`KaiRawal/Rule-of-Thumb`) is the standalone home of
`ruleofthumb`, a pip-installable Python package consolidating three
variants of the RoT explainer into one library:

- **Tabular** (`ruleofthumb.explain.RuleOfThumb`) — vector inputs, one
  importance weight per feature.
- **Text / LLM embeddings** (`ruleofthumb.text.RoTText`,
  `ruleofthumb.explain.TextRuleOfThumb`) — token-by-embedding inputs with
  explicit padding masks and length-normalised scores.
- **Images** (`ruleofthumb.image.RoTImage`) — importance shared across spatial
  locations of feature maps.

The package explains a black-box model by fitting a transparent surrogate on
partial observations of its behaviour, producing per-feature importances that
can be revealed incrementally (most-important-first) while preserving the
black box's predictions.

Provenance: ported from the research monorepo's `pip-package` branch. Terse
`(source: legacy ...)` pointers in ToDo.md are provenance notes only — the
legacy code did not move with the package and never constrains new work.
When a ToDo item specifies new functionality, the ToDo text defines the
design and wins. Notable departures from legacy behaviour deserve a sentence
in the changelog entry.

## Environment rules

- All development happens in **`.venv`** at the repo root (editable install
  of the package + `[dev]` dependencies).
- Run everything from the repo root:
  - `.venv/bin/python -m pytest`
  - `.venv/bin/python -m ruff check .`
  - `uv build`
  - `.venv/bin/jupyter nbconvert --to notebook --execute --inplace examples/<nb>.ipynb`
- This machine runs **lightweight verification only**: the test suite (~1s),
  lint, sdist builds, and executing the small example notebooks. Do not run
  heavy training or experiments here.

## Package architecture

```
.
├── pyproject.toml          # uv_build; single-install deps; [dev]/[docs] extras only
├── LICENSE               # MIT licence
├── .gitignore
├── mkdocs.yml + docs-src/  # docs source (guides, notebooks, API, test report)
├── docs/                   # built site, committed (Pages serves main's /docs)
├── PUBLISH.md              # PyPI + Pages release runbook (do not publish from CI)
├── requirements.txt        # pinned env matching pyproject (convenience)
├── ToDo.md                 # forward-looking package TODOs + changelog footer
├── README.md               # user docs incl. v0.1→v0.2 migration guide
├── src/ruleofthumb/
│   ├── core.py             # RoT base class
│   ├── text.py             # RoTText, pad_sequences, sentinel_mask, lengths_to_mask
│   ├── image.py            # RoTImage, pad_images
│   ├── explain.py          # Explainer facade + fit / fit_tabular / fit_text / fit_image factories
│   └── __init__.py         # exports + __version__
├── tests/                  # pytest suite (test_masks.py pins mask/reveal behaviour;
│                           #   tests/integration/ runs against committed real-data/model
│                           #   artifacts — see tests/integration/generate_artifacts.py)
└── examples/               # hello-world quickstart notebooks (dummy data)
```

Key mechanics:

- `RoT.training_loop` runs an SGD loop with SWA averaging (SWA model stored via
  `object.__setattr__` to avoid submodule registration — required for
  torch >= 2.x).
- The incremental-reveal pipeline (`get_order` → `ordered_predict` →
  `score_ordering`) simulates revealing inputs most-important-first and scores
  prediction fidelity along the curve. Reveal units are tokens (text) /
  pixels (images) / features (tabular) by default.

## Design decisions (binding)

These were deliberate choices made during the v0.1 → v0.2.x evolution. Do not
regress them:

1. **Masks are first-class and explicit** (v0.2.0, breaking change). Padding
   is *never* inferred from data values; `-1` has no special meaning anywhere.
   Convention: masks are boolean validity tensors (`True` = real token /
   pixel). Utilities return `(padded_batch, lengths_or_mask)`.
   `ruleofthumb.text.sentinel_mask` exists solely to migrate legacy `-1`
   padded arrays.
2. **Unit-granularity reveal curves by default** (v0.2.1): one reveal step per
   token (text) or pixel (image), aggregating embedding dims / channels via
   abs-sum. `granularity="element"` restores per-feature-element curves; the
   granularity used by `ordered_predict` / `score_ordering` must match how
   the order was produced (no auto-detection). Tabular is unaffected by the
   setting.
3. **PEP8 class names without aliases**: `RoTImage`, `RoTText`. Breaking API
   changes are acceptable pre-1.0, but each one needs a README migration note.
4. **Single install**: all runtime dependencies live in base `dependencies`
   (plotting, transformers, vision included); only `[dev]` exists as an
   extra. License is SPDX `license = "MIT"`.
5. **Dead code is deleted**, not kept "for port fidelity". Removals are
   recorded in the ToDo changelog footer instead.
6. **Known deliberate limitations** stay until their ToDo item lands:
   `classes=2` hard-coded, binary-only reduction metrics, class-level
   `mins`/`maxs`, hard-coded training hyperparameters (5 pretrain epochs,
   SWA burn-in `epochs // 10 + 1`, weight decay `0.01`, `l1_penalty 0.01`).
7. **Outputs honour the explicit-mask contract** (v0.2.13). Padding /
   embedding utilities may normalise their outputs to "the mask carries the
   truth": e.g. `embed_texts` writes zeros into padded rows even though
   transformers emit non-zero hidden states there. Downstream models exclude
   masked positions either way; normalised outputs are simply predictable to
   inspect.

## ToDo.md conventions

`ToDo.md` lists **forward-looking improvements to the pip package only**.
Structure:

- *API / correctness* — refactors and generalisations of existing APIs.
- *New functionality* — new modules and capabilities (e.g. native ingestion
  of raw strings / image file paths, plotting, additive shape functions).
  Terse legacy source pointers like `(source: legacy rot_class.py)` are
  allowed as optional provenance notes.
- *Non-goals* — explicit design decisions against features (do not re-litigate
  them silently).
- *Changelog footer* — completed work summarised per version; once an item
  ships it moves from the todo list to here. The footer is ordered strictly
  newest-first, with each new entry inserted directly under the `## Changelog`
  heading.

Item numbers are permanent identifiers: completed items simply leave gaps in
the numbering. Never renumber, compress or shift remaining items.

When behaviour changes, update README (usage + migration notes) and bump the
version.

## Definition of done

Before finishing any change:

- [ ] `pytest` green, `ruff check .` clean (run inside `.venv`).
- [ ] Integration tier (`tests/integration/`) stays green and artifact-based:
      black boxes are never trained and datasets never downloaded at test
      time. If artifacts must change, regenerate them with
      `tests/integration/generate_artifacts.py` (GNU `timeout`-guarded) and
      commit them together with the code change; the whole suite should stay
      under ~2 minutes. Only `make_pets` needs the research monorepo
      checkout (it raises a clear error without it); everything else
      regenerates standalone.
- [ ] `uv build` succeeds if packaging changed.
- [ ] Touched example notebooks still execute end-to-end without errors.
- [ ] Version bumped in all three places when behaviour changed:
      `pyproject.toml`, `src/ruleofthumb/__init__.py`, and the assertion in
      `tests/test_explain.py`.
- [ ] Untracked caches cleaned up: `__pycache__/`, `.pytest_cache/`,
      `.ruff_cache/` (all gitignored, but keep the tree tidy).
