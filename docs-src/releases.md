# Release notes

What changed in each release, newest first. Breaking changes across
the 0.x line are recorded in [Migration notes](migration.md); the
full per-version changelog lives in `ToDo.md` in the repo. The
release ritual itself (bumps, tags, TestPyPI/PyPI, docs rebuild) is
documented in the [Development guide](development.md).

## 0.0.2

Breaking padding-API cleanup plus robustness hardening over 0.0.1:

- Text/image entry points speak a single `mask=` spelling — the
  `lengths=` / `attention_mask=` kwargs are gone (build masks with
  the newly exported `lengths_to_mask`).
- New `ruleofthumb.vision` module (default backbone for image file
  paths); `embed_texts` gains a pinned `revision=` default.
- `autotune` forwards `n_classes` and extra factory kwargs; fits
  warn on oversized batches and low train agreement.
- Explainer save files load only under the exact package version
  that wrote them.

## 0.0.1

First public pre-alpha (version reset from internal v0.2.19 to
reserve the PyPI name `ruleofthumb-rot`). Full documentation site,
GitHub Actions (checks, TestPyPI on `main`, tag-gated PyPI via
trusted publishing), type annotations on the public functions.
Expect breakage and backwards-incompatible changes before any 1.0 —
see [Migration notes](migration.md) and the
[install](install.md) caveat before depending on it.
