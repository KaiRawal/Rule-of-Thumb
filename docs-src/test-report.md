---
orphan: true
---

# Test-suite report

The full report is maintained at `tests/TEST_SUITE.md` and included here
verbatim so the site always matches the tree. Its maintenance rule:
adding, removing, renaming, or re-thresholding any test must update the
report's exhaustive table.

## How to read it (and how to compare)

Every accuracy number below is **fidelity of the stand-in to the
black box**, not accuracy against ground truth — that is the number
RoT is responsible for, and the one the floors assert. Three
companions keep each number honest:

- **Majority baseline.** The always-guess-the-commonest-class score
  sits next to most measurements. A fidelity near majority on a
  hard task (raw-image 10-class) documents a capacity ceiling, not
  a failure — see [Limits](capacity.md).
- **Random-order controls.** Reveal curves and human-agreement
  scores are reported against random baselines (random uncovering
  order, random importances). An explainer earns its keep only by
  beating them; the [HateXPlain](notebooks/04_hatexplain.ipynb) and
  [gaze](notebooks/05_salicon.ipynb) notebooks show the comparisons
  live.
- **Human-agreement scores.** Weighted AUROC against rationale
  spans, pointing accuracy, box overlap — reported alongside the
  box's own numbers and, where computed, exact-SHAP parity. These
measure plausibility, a separate judgement from fidelity (see
[Core ideas](concepts.md)).

Comparing against another explainer? Match the setup first: same
black-box outputs, same inputs, same split — then compare fidelity
floors, reveal-curve gaps over random, and human-agreement scores
in that order. Runtimes in §6 are machine-specific baselines for
spotting regressions, not benchmarks to beat.

```{include} ../tests/TEST_SUITE.md
```
