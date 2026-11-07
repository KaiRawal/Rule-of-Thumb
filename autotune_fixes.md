# Autotune fixes (ToDo 37, unblocks ToDo 36)

Changes to `autotune` so it can reach the hyperparameters that reproduce the
monorepo papers on its own, instead of needing them hand-set per experiment.

## Why

The ToDo 36 replication report found autotune never reproduced a paper target:

- **Search space too narrow and one-size-fits-all (gap #7).** A single flat
  space (`learning_rate [0.003..0.1]`, `batch_size [64, 500, 2000]`,
  `epochs [100, 300, 600]`) excluded every setting that reproduces the papers:
  text needs learning rates down to `1e-6` and batches of 16-32, tabular needs
  `0.005` and up to 1500 epochs, and `l1_penalty` was not searchable at all.
- **One bad candidate killed the whole search (gap #16).** A candidate that ran
  out of memory on long text aborted the run before it found anything.
- **Ties were not deterministic (gap #8).** On easy problems many candidates
  tie; the first one sampled won.

## What changed (`src/ruleofthumb/tune.py`, `src/ruleofthumb/explain.py`)

1. **Per-modality default search space** (`DEFAULT_SPACES`). Each modality gets
   ranges suited to it; `l1_penalty` is searchable for text only (the only
   factory that accepts it). The space validation and the `tune=True` wrapper
   both use the modality's space.

   | knob | tabular | text | image |
   |---|---|---|---|
   | learning_rate | 0.005, 0.01, 0.05, 0.1 | 1e-6, 1e-5, 1e-4, 0.01 | 0.003, 0.01, 0.05 |
   | batch_size | 64, 500, 2000 | 16, 32, 64 | 500, 2000, 5000 |
   | epochs | 100, 500, 1500 | 35, 100, 300 | 100, 300, 600 |
   | weight_decay | 0.0, 0.01, 0.05 | 0.0, 0.01 | 0.0, 0.01, 0.05 |
   | l1_penalty | - | 0.0, 0.01 | - |

2. **OOM-only candidate skipping.** A candidate that runs out of memory (CUDA or
   CPU) is recorded (`score = -inf`, with the error in `trial["error"]`) and
   skipped; the search continues. Any other error (bad kwargs, out-of-range
   labels) still propagates, so real bugs fail fast. If every candidate OOMs,
   `autotune` raises a clear `RuntimeError`.

3. **Deterministic tie-break.** Equal scores break toward the cheaper fit
   (fewer epochs, then larger batch).

## Tests (`tests/test_tune.py`)

Added: `test_default_space_is_modality_aware`,
`test_oom_candidate_is_skipped_and_search_survives`,
`test_non_oom_error_propagates`, `test_all_candidates_oom_raises`.

All 17 `test_tune.py` pass; `ruff check` clean on the touched files. (Two
unrelated `test_explain.py` failures exist on this branch regardless of this
change, from newer library versions in the test environment.)

## Not done here, and why

- **Scoring still optimises agreement, not the paper's explanation targets
  (gap #4).** There is no well-defined explanation metric to optimise instead,
  so this is left as a known limitation rather than guessed at.
- **Cost control / plateau stopping / early abandonment (ToDo 37 item 29).**
  Touches the training loop; a separate, heavier change.
- **Text memory fix (gaps #3/#4).** An `einsum` token-level loss and not copying
  the whole dataset to the GPU live in `text.py`, not autotune. The space now
  offers small batches, but Judicial still needs this fix to run at the paper's
  batch sizes without OOM.
- **The `b`-start decision (ScientificDiscovery, OpenXAI).** A code/design call,
  not a tunable knob; nothing here reaches it.

## Effect

Autotune can now reach the paper-reproducing hyperparameters without hand-setting
them. That is the autotune half of recovering Movie and Judicial for ToDo 36; the
text-memory fix and the `b` decision are what remain for those.

## ToDo 29 — fit-to-convergence / early abandonment

Done: **successive halving** in `autotune`. With four or more candidates the search screens every
candidate at ~1/3 of its epochs, ranks them, and finishes only the better-scoring half at full epochs;
abandoned candidates keep their cheap screen score (marked `pruned`). This is the "give up on hopeless
hyperparameter combinations early" half of ToDo 29 and the main affordability win, and it lives entirely
in `tune.py`.

Deferred (on purpose): the per-fit **plateau stopping inside the training loop** (stop when agreement
plateaus, with a max-epoch cap). That touches `RoT.training_loop`, which runs SWA averaging over its tail
epochs, so an early stop there changes every fit's numerics and would disturb the committed weights and
test floors (ToDo 22). It is P1, so it is left as a follow-up rather than risking the whole suite for it.
