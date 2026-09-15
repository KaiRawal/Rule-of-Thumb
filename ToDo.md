# ruleofthumb — ToDo

Planned improvements to the pip-installable `ruleofthumb` package, ordered
into execution tiers: quick wins first, then foundational API work that later
features build on, then new functionality.

Standalone home: this package now lives at
`https://github.com/KaiRawal/Rule-of-Thumb` (PyPI `ruleofthumb`, docs at
`https://ruleofthumb.readthedocs.io/`), ported from the research
monorepo's `pip-package` branch. Terse `(source: legacy ...)` pointers below
are provenance notes only; the legacy code did not move with the package.

## API / correctness

22. **Committed-weights robustness program for the test suite (P0).** Long SGD
    fits in underdetermined regimes land in different minima per BLAS
    kernel, so no live-fit absolute floor is portable across hosts (seen:
    pets heatmap corr `0.36–0.52` vs `1.0`, text native accuracy `0.65`
    vs `1.0`, digits top-2 `0.80` vs `0.92`). Direction (decided): the
    testing pyramid splits by *what is asserted*, not by tier. Unit tests
    that assert on the *outputs* of a pre-fit RoT (explanations, plots,
    persistence round-trips, mechanism recovery) load raw `.pt` state
    dicts as mock fitted models and assert tight margins; integration
    tests keep fitting live end-to-end with honest loose floors plus
    structural checks, hardened by bigger-N artifacts and proxy-ensemble
    floor setting (see below). Converted unit sets (`tests/fixtures/`,
    minted by `tests/mint_unit_weights.py`, loaded via
    `tests/_mock_weights.py`): plot rendering (tabular binary +
    multiclass), persistence round-trips (tabular/text/image, fit
    incidental), calibrated mechanism recovery (binary + multiclass),
    faithfulness probes (deletion/pointing/noise), nonlinear ring
    separation (linear + rbf + hinge). Must stay live everywhere:
    seed-reproducibility, tune search, native-ingestion fits, comparative
    fits (including the linear-vs-rbf integration pair). Raw `.pt`, never
    `.rotx` (version-stamp coupling; `.rotx` stays covered by the
    round-trip tests). Dataset builders are single-sourced in the loader
    so mint fits and test inputs cannot drift. Integration hardening, in
    order: (1) determinism (`torch.use_deterministic_algorithms` where
    ops allow, batch-order pinning, residual-BLAS note in
    `development.md`); (2) bigger-N artifacts (digits tabular 500→1797,
    COMPAS 800→2000, reviews 31→~60 hand-written, image digits 500→1000+;
    breast-cancer/wine are full-size already); (3) proxy-ensemble floor
    setting (seeds × threads × device, floors at ensemble minimum minus
    margin, recorded in `TEST_SUITE.md`). Regen discipline in
    `development.md`, `TEST_SUITE.md` updates. Done when CI is green with
    tight unit margins on mocks and measured ensemble-backed floors on
    live integration fits.

23. **RoT stability across hardware (research, P2).** The suite has proven the
    method property behind item 22: long fits in underdetermined regimes
    (N=31/E=768 text, 20-sample/576-ch pets) reach different minima per
    kernel, so explanations and fidelity can wobble across machines.
    Characterize cross-kernel variance, evaluate mitigations (early
    stopping, stronger regularization, weight averaging, ensembles),
    cross-referencing item 16's capacity ceilings. Success is
    explanations/fidelity invariant across kernels; any API change needs
    a migration note.

24. **Explanation quality beyond fidelity (research, P1).**
    Fidelity-to-black-box and plausibility-to-human-rationales are
    separate metrics and must be measured separately: deletion/insertion
    curves always against random baselines, rank-agreement metrics, and
    mandatory trivial controls (position/center priors, which beat
    learned maps on human-agreement scores). Short or noisy inputs defeat
    pooled representations for rare lexical signals (further evidence
    for item 16); near-single-class black boxes need explicit handling
    before any explanation is attempted. No implementation commitment.
25. **Center-prior comparison for image agreement (research, P2).**
    Human-agreement scores for saliency are dominated by position priors
    (a static center control beats learned maps). The fidelity gate and
    any saliency-as-detector guidance should require beating a center
    control by margin, or the docs must state plainly that saliency is
    not gaze prediction. No implementation commitment.
28. **Allow non-0.5 dropout rates (done: 8a9b902).** Dropout is currently a fixed method
    constant (0.5): the constructors, factories and `autotune` search
    space accept no `dropout_rate`, since only 0.5 keeps every partial
    observation equally likely (uniform over reveal subsets). If the
    method is ever generalised to biased reveal distributions, re-open
    the parameter behind a flag and reintroduce the search dimension —
    with reveal-curve scoring reweighted to match the sampling
    distribution. Until then the `test_dropout_rate_is_not_tunable`
    pin stays (pinned at `tests/test_tune.py:186`).

## New functionality

16. **Optional per-location importance weights (deferred, P3: rich backbones cover
    real use).** Definitions: `K` is the number of classes (2 for cat-vs-dog,
    10 for digits); `E` is the embedding length per token; `C` is the channel
    count per pixel; `T`/`H`/`W` are tokens/height/width. The text and image
    variants share their `a` / `b` parameters across all positions — shape
    `(K, E)` for text, `(K, C)` for images (`2·K·C` parameters, independent of
    spatial size) — making each surrogate a linear model on a pooled
    representation: the token-mean embedding for text, per-channel spatial
    sums for images. This is faithful to the legacy design and yields
    position-invariant saliency, but caps fidelity at the separability of the
    pooled vector: raw single-channel images pool to total ink mass alone, so
    multiclass fidelity floors out near the majority baseline (always guessing
    the commonest class) no matter how long the fit runs (measured on digits;
    see `tests/integration/test_image_integration.py`, which pins the raw
    failure, the coordinate-channel partial recovery to ~0.35, the TinyCNN
    trunk at only ~0.49, and the strong MobileNetV3-Small recovery to ~0.99
    with 500 images exceeding 49 spatial positions). The same applies to
    text: pooling to the token-mean makes word order invisible, so tasks
    driven by syntax, negation scope or token counts — rather than which words
    appear — hit an analogous ceiling even with high-dimensional embeddings,
    while lexical tasks and rich pretrained embeddings stay strong. Real-life
    users should use rich backbones (transformer embeddings, MobileNet
    feature maps) and check surrogate predicted-class accuracy; only if
    raw-input multiclass is required, consider an opt-in unshared mode —
    image weights `(K, C, H, W)` / text weights `(K, T, E)` or per-position
    biases — turning the surrogate into a full linear multiclass model over
    raw inputs (`2·K·H·W` parameters), behind a constructor flag so the
    legacy behaviour stays the default.

20. **Documentation content and presentation pass (P1).** The reference docs are
    complete but the site undersells the package: no conceptual overviews
    (when RoT fits, how it differs from sensitivity-based explainers, how
    to judge trust), examples are hello-worlds on dummy data only, plots
    have no visual gallery, and the test-suite report isn't framed for
    comparison. Content work, in order: (a) overview essays; (b) real-task
    example gallery per modality built from the integration artifacts;
    (c) per-plot-type visual gallery; (d) benchmarks framing; (e)
    release-notes and contributing pages surfaced from the existing
    runbooks. Presentation work within the current theme: logo/favicon,
    landing hero image, announcement banner for the pre-alpha caveat,
    gallery thumbnails, tighter nav grouping, small custom CSS. Non-goal:
    no theme switch.

21. **Publication-ready plots via `shap-editorial` (P1).** `shap-editorial`
    (MIT, v0.1.x alpha, single maintainer — exact pin, treat as
    replaceable) renders beeswarm/waterfall/bar/scatter from duck-typed
    `shap.Explanation` objects, returning `(fig, ax)`; it is now a base
    dependency so it can be imported anywhere in the package. Our tabular
    path already packs RoT values into `shap.Explanation`, so spike
    compatibility first: if the objects are accepted as-is, expose
    editorial variants (`plot.editorial_*` or a `style=` flag) starting
    with tabular waterfall/bar/beeswarm. Explicit decisions required:
    palette (editorial grey→red clashes with our binding red=toward /
    blue=against — override to RoT convention or adopt editorial
    deliberately, recorded in the changelog); multiclass (rejected
    upstream — slice a class first, matching our per-class outputs). Ship
    with tests, a gallery demo under item 20(c), and README/docs updates.

27. **Full-dataset and large-scale validation (research, P3).** The
    integration tier validates on slices (3000-post text slice,
    150-subset images) because the dev machine cannot stage dense
    full-train embeddings (~4.5 GB float32 for 15k×96×768) or finish
    long fits quickly. Two tracks to close the gap: (1) verify the
    human-annotation benchmarks on a large machine, using GPU/MPS and
    threading speedups, and record the slice-vs-full deltas; (2)
    implement out-of-core RoT training — stream batches from disk
    (memmap chunks) through `training_loop` so fits run when the data
    does not fit in memory at once. No implementation commitment.
29. **Keras-style fit-to-convergence training (P1).** Searching a fixed
    `epochs` grid is a stand-in, not a stopping rule: train until the
    surrogate is determined to be fit — stop when surrogate-vs-blackbox
    agreement (or validation reveal fidelity) plateaus, with a max-epoch
    cap — and let `autotune` give up on hopeless hyperparameter
    combinations early (successive-halving style: abandon candidates
    whose early fidelity trails the pack instead of running every
    candidate to completion). Cross-refs item 16 (capacity ceilings bound
    what "fit" can mean) and item 23 (stability across minima).

30. **Opt-in explanation evaluation (agreement, reveal gap, human-data
    comparison) (P1).** An explicit `Explainer.evaluate(...)` reporting only
    realised, research-backed measures: (a) **fit agreement** —
    in-sample plus a seeded held-out split, warning below 0.75
    (extends the existing `train_agreement_` tripwire rather than
    replacing it); (b) **reveal gap** — reveal-curve AUC minus a seeded
    random-order AUC via the existing `score_ordering`, computed only
    on explicit request and off by default; (c) **human-data comparison
    iff supplied** — length-weighted AUROC of importances against
    user-provided rationale masks with a random-importance baseline
    alongside (movie-review/judicial practice: per-span scores,
    sign-flipped toward the predicted class), text-first and
    mask-aware. Shipped defaults this item records: `autotune` scores
    by plain held-out agreement unless `scoring="reveal"` is named,
    and fits never run reveal machinery (only `predict` for the
    agreement tripwire). Explicitly out of scope: seed-stability
    refits, slice breakdowns, center priors, new dependencies, and
     black-box insertion scoring — the package never holds the model,
     only its labels, so interventions stay an opt-in accuracy-testing
     path, never a default.

31. **Per-datapoint misprediction warning (pending, P1).** When a user
    explains rows whose model predictions are known, rows the
    stand-in itself gets wrong deserve a warning: their explanations
    are absurd and meaningless, and the user should be told so —
    without any error or blocking path. Design: an optional
    `y_outputs=` kwarg on `Explainer.get_explanation` (the method
    otherwise never sees black-box answers, so nothing changes unless
    the kwarg is passed); on mismatch, a single `UserWarning` per
    call naming the count and row indices (capped list). Plot helpers
    call `get_explanation` internally without labels and stay silent;
    `get_order` never computes explanations and is out of scope.
    Length mismatches fail fast with `ValueError`, matching the
    file's existing style. Tests: mismatched rows warn with correct
    indices, matching rows stay quiet, plus multiclass and masked-text
    cases. Not started.

32. **Brand/graphics refresh (static assets only, P2).** Distinct from
    improving the `plot.*` visualisations themselves and from item
    20's in-theme presentation work: redraw the repo/docs identity
    set — mark (`logo.svg` / `logo-dark.svg`), favicon set, README
    header banner, and hero/card figures regenerated through
    `docs-src/_figures/generate_figures.py` (the dummy-data-only rule
    stays; light/dark pairs stay). Scope is `_static/` plus the
    `README.md` header and the `index.md` hero/cards only: no
    `plot.py` API change, no theme switch (item 20's non-goal
    stands), and the palette keeps the binding red=toward /
    blue=against convention. Done when the assets land in pairs and
    every referencing page (`index`, `plots`, `README`) renders in
    both themes with a green RTD build. Not started.

35. **Interactive explanation explorers (spike, tech open, P2).** No
    interactive visualisation exists today (repo-wide grep: no
    `ipywidgets` / `anywidget` / `plotly` / `bokeh`; `plot.text_html`
    is static HTML, everything else returns a static matplotlib
    `Figure`). Goal only, stack to be decided by spike: prototype
    `anywidget` custom views *and* `ipywidgets`-driven matplotlib
    controls against the same computed `imp` / `order` (reveal-step
    slider, saliency `power` / `trim` sliders, token hover), then
    pick per effort, fidelity and RTD-embeddability. Hard
    constraints: a new optional extra only, the `plot.*`
    Figure-returning contract unchanged (widgets wrap, never
    replace), a static fallback always one call away, and no new
    required dependencies. Ship the decision plus one pilot widget
    with tests, a gallery demo under item 20(c), and README/docs
    updates. Done when the pilot renders in JupyterLab and VSCode
    (plus Colab/RTD if the chosen stack allows). Not started.

## Release / maintenance

18. PyPI release checklist: three-place version bump (`pyproject.toml`,
    `src/ruleofthumb/__init__.py`, `tests/test_explain.py` assertion) plus
    `ToDo.md` changelog entry, then commit, push, and tag — CI builds and
    uploads (TestPyPI on `main`, PyPI on `v*` tags via trusted publishing),
    RTD rebuilds; verify each stage per PUBLISH.md. (Standing ritual —
    last exercised at release 0.0.2, `b4c32d4`.)

33. **Conda-forge distribution alongside PyPI (P2).** Conda-forge is
    currently absent everywhere (no recipe, no workflow mentions).
    Work: provision a `ruleofthumb-rot` feedstock via a staged-recipes
    PR (conda-forge name to confirm against the PyPI
    `ruleofthumb-rot` vs import-name `ruleofthumb` mapping), with a
    `meta.yaml` mapping the base and extras dependencies to conda
    packages (verify torch CPU builds plus `transformers`,
    `torchvision`, `captum` and `shap` availability; `noarch: python`
    only if everything resolves, else an arch split). Thereafter rely
    on the regro-cf autotick bot watching PyPI for version bumps
    rather than custom automation, and extend item 18's checklist
    plus `PUBLISH.md` with the feedstock step. Done when `mamba
    install -c conda-forge ruleofthumb-rot` works from a clean env
    and the bot picks up the next tag without manual intervention.
    Standing cost, recorded here: feedstock maintenance (pin churn,
    bot-PR merges). Not started.

34. **Binder launch for all six example notebooks (P3).** Add repo2docker
    config (`environment.yml` with mamba plus `postBuild`) and
    launch badges (`README.md`, `docs-src/examples.md`) covering the
    dummy-data hello-worlds (`01`, `02`, `03`, `06`) *and* the heavy
    `04_hatexplain` / `05_salicon` notebooks, with explicit
    mitigations rather than rewrites: the env must fit a shared
    Binder session (torch plus `transformers`, `torchvision`,
    `captum`, `shap` make a GB-scale image with slow cold builds),
    the `04` / `05` download cells (dataset JSONs, MIT zips, model
    weights) stay as-is but slow first launches get documented, and
    the ~2 GB RAM ceiling is validated by actually launching each
    notebook top-to-bottom from the badge. `mybinder.org` itself is
    free (no account, no SLA, queued shared builds). Non-goal: no
    notebook rewrites to fit Binder; the RTD-executed docs stay
    canonical. Done when all six run badge-to-finish in a fresh
    Binder session. Not started.

## Non-goals

- Bitstring/partial-information sampling scripts are not needed by design:
  RoT operates per token/pixel.

---

## Changelog

- **v0.0.2** — breaking padding-API cleanup plus robustness hardening over
  the published v0.0.1. Breaking: text/image entry points speak a single
  `mask=` spelling — the `lengths=` / `attention_mask=` kwargs are gone
  (build masks with the newly exported `lengths_to_mask`) and the
  `sentinel_mask` migration helper is deleted. New: `ruleofthumb.vision`
  module (`embed_images` / `ImageEmbeddings` / `DEFAULT_IMAGE_MODEL`,
  the default backbone for image file paths); `embed_texts` gains
  `revision=` (pinned to the new `DEFAULT_TEXT_REVISION` by default);
  `autotune` forwards `n_classes` and extra factory kwargs to candidates
  and the final refit. Behaviour: out-of-range labels fail fast with a
  clear error, inference entry points always return host-side results,
  oversized batches and low train agreement (`train_agreement_`) warn,
  read-only input arrays are accepted silently, and explainer save files
  load only under the exact package version that wrote them. No change to
  the linear surrogate itself; omitting the new arguments keeps prior
  behaviour.
- **v0.0.1** — first public pre-alpha (version reset from internal v0.2.19
  to reserve the PyPI name (`ruleofthumb-rot`) and invite early feedback). Entirely vibe-coded
  from hand-written research code; expect breakage and
  backwards-incompatible changes before any 1.0. Build backend switched
  from hatchling to `uv_build`; package metadata completed (`LICENSE`
  vendored, `license-files`, `project.urls` incl. Documentation,
  `Development Status :: 1 - Planning`, `[docs]` extra). Full
   documentation website (closes item 19): modality guides with executed
   notebooks, workflows, migration notes, capacity guidance, per-module API
   reference, and the test-suite report — built with one command
   (`sphinx-build -W`), hosted on ReadTheDocs (versions + previews). GitHub
   Actions added (item 17): checks (pytest/ruff/sphinx) on push/PR,
   TestPyPI on `main` pushes, tag-gated PyPI releases via trusted
   publishing; ritual documented in `docs-src/development.md`. Type
   annotations added to the five public
  functions griffe flagged (`embed_texts`, `autotune`, `load_images`,
  `load_explainer`, `plot.saliency`); behaviour unchanged. No behaviour change.
- **v0.2.19** — opt-in non-linear additive explanations: every factory and
  RoT constructor accepts `nonlinear=` (a string — `"rbf"` Gaussian bumps or
  `"hinge"` SELU hinges — or a dict `{"type": ..., ...}` whose extra keys are
  forwarded as hyperparameters, e.g. `{"type": "rbf", "n_bases": 32}`). The
  learned elementwise response `s` turns the surrogate into
  `imp[k,i] = a[k,i]·(s(x[i]) + b[k,i])`: per-feature non-linear shape
  curves shared across all input elements (parameter budget independent of
  input size), for all modalities and class counts. Both responses are
  residual with zero-initialised coefficients, so unfitted non-linear models
  are exactly the linear ones; explanations stay exactly additive per
  feature element, so plots, reveal curves and persistence work unchanged,
  and the configuration round-trips through save files. Default behaviour is
  unchanged: omitting `nonlinear` keeps the plain linear model.
- **v0.2.18** — new `ruleofthumb.plot` module. Tabular: SHAP's signature
  plots (waterfall, force, decision, bar, beeswarm) rendered by delegating
  to the `shap` package (new base dependency) with RoT values packed into a
  `shap.Explanation`; the SHAP base value maps to the RoT class bias `g_k`
  and `f(x)` to the surrogate score (documented in the README's baseline
  semantics table). Text: token highlighting as IPython-aware HTML and a
  static matplotlib export, plus aggregated positive/negative/combined word
  clouds (red = toward the class, blue = against — departing from the legacy
  clouds' green/red for consistency with the legacy saliency overlays).
  Images: port of the legacy saliency overlay (sign-preserving power
  transform, independent percentile-trimmed normalisation, saturation
  compression) over optional RGB input. All functions return matplotlib
  figures or HTML; nothing auto-shows.
- **v0.2.17** — automatic hyperparameter tuning: `ruleofthumb.autotune`
  searches `learning_rate` / `batch_size` / `epochs` / `dropout_rate` /
  `weight_decay` (grid or seeded random search over a customisable space,
  default `DEFAULT_SPACE`) with a seeded validation split; candidates are
  fitted through the regular factories with per-candidate seeds and scored
  on held-out data by final-step reveal accuracy. Returns an `AutotuneResult`
  whose `.explainer` is the winning configuration refit on all data, plus
  `.best_params`, `.best_score` and best-first `.trials`. Supports all three
  modalities including native string / file-path inputs.
- **v0.2.16** — explainer persistence: `Explainer.save(path)` writes a
  versioned payload (model `state_dict`, per-modality constructor config,
  `mins` / `maxs`; primitives and tensors only, loadable with
  `weights_only=True`) and `ruleofthumb.load_explainer(path, device=...)`
  reconstructs the explainer without refitting — explanations, predictions
  and reveal-curve outputs are identical after the round-trip. RoT models now
  record their construction `sample_shape` (needed to rebuild text models,
  whose `(T, E)` shape is not recoverable from the weights alone). Native
  string / file-path ingestion is not persisted; reloaded explainers consume
  numeric arrays.
- **v0.2.15** — native image ingestion: `fit_image` (and `fit`'s
  auto-detection, which routes image file extensions to the image modality)
  accept image file paths directly; files are decoded with Pillow (RGB,
  `[0, 1]` floats) either at native sizes (zero-padded with derived validity
  masks) or resized + centre-cropped via `size=`, and `transform=` replaces
  the whole pipeline for caller-supplied preprocessing (e.g. torchvision
  weights transforms). New `ruleofthumb.image.load_images` /
  `ImageBatch` mirror the text-side utilities. Explainers fitted from paths
  accept the same paths back in every public method — each call re-loads the
  files.
- **v0.2.14** — native text ingestion: `fit_text` (and `fit`'s auto-detection)
  accept raw strings directly, embedding them with the bundled
  `answerdotai/ModernBERT-base` default and deriving attention masks
  automatically; callers supply `tokenizer=` / `model=` to override the
  embedder. Explainers fitted from strings accept the same strings back in
  every public method (`get_explanation`, `get_order`, `ordered_predict`,
  `score_ordering`, `score`, `predict`) — each call re-embeds the texts;
  string inputs compose with no explicit padding arguments.
- **v0.2.13** — new `ruleofthumb.embed` module: `embed_texts` tokenises and
  embeds raw strings with a HuggingFace transformer (default
  `answerdotai/ModernBERT-base`, overridable via `tokenizer=` / `model=`),
  returning a frozen `TextEmbeddings` dataclass with rectangular zero-padded
  `(N, tokens, dim)` float32 embeddings, a boolean attention mask ready for
  `fit_text`, and decoded per-sample token strings aligned with the embedding
  rows; batching, `max_length=` truncation and `device=` (auto-detect
  cuda > mps > cpu) supported. Port of the legacy `gen_token_embeddings.py`
  workflow as a library function.
- **v0.2.12** — integration tier expanded and hardened; every case now asserts
  the RoT surrogate's own **predicted-class accuracy** against its black box
  plus explicit feature-importance anchors: breast-cancer explanations track
  the LogisticRegression coefficient profile, COMPAS explanations rank
  `priors_count` top for GBM/SVC/MLP black boxes, wine models agree on shared
  dominant features, text top tokens carry sentiment words (with a
  "brilliant" > "awful" pair check), pet saliency maps reproduce committed
  reference heatmaps and point in the dog direction for GPT-"dog" images, and
  the 10-class image confusion matrix is asserted to collapse near the
  majority baseline (documenting the spatially-shared-weight capacity limit).
  New cases mirror real DS workflows via pandas: the legacy GPT-4o-mini
  cat-vs-dog experiment miniaturized (raw JPEGs + labels committed from
  `ExplanationExampleRemote/DATA` read-only; MobileNetV3-Small features and
  all embeddings recomputed afresh every run — never cached), COMPAS
  two-year recidivism fetched once by the generator with the canonical
  ProPublica filters, and sklearn wine. Typical black boxes added:
  GradientBoosting/SVC(RBF)/MLP per dataset. `pandas` joined the `[dev]`
  extra.
- **v0.2.11** — integration-test tier (`tests/integration/`) running against
  real data and models: breast-cancer + LogisticRegression (tabular binary),
  digits + RandomForest (tabular multiclass), fixed film reviews +
  distilbert-SST-2 (text binary), and digit images through committed TinyCNN
  black boxes (image binary and 10-class). All black-box models/datasets are
  built once by `tests/integration/generate_artifacts.py` and committed under
  `tests/integration/artifacts/` (with a provenance `manifest.json`); the test
  suite trains nothing and downloads nothing except HF-cached SST-2 weights.
  New unit tests cover the remaining cells: multiclass text (reveal pipeline,
  per-class padding, mask equivalence, seeding) and binary images via the
  facade with a conv black box. `scikit-learn` moved to the `[dev]` extra —
  it is only needed to (re)generate artifacts, never at runtime.
- **v0.2.10** — unified explainer facade (breaking): the `RuleOfThumb` /
  `TextRuleOfThumb` wrapper classes are replaced by the public `Explainer`
  class plus `fit` / `fit_tabular` / `fit_text` / `fit_image` factories
  (`fit` auto-detects the modality from input ndim). Adds the previously
  missing image wrapper (signed per-pixel explanations, channels summed,
  `(N, H, W)` masks) and public delegating reveal-pipeline methods
  (`get_order`, `ordered_predict`, `score_ordering`, `score`, `predict`);
  raw models are unchanged.
- **v0.2.9** — `device=` parameter on all three RoT models and both explainer
  wrappers (`None` auto-detects cuda > mps > cpu; default cpu otherwise).
  Fit and inference move inputs to the model's device; raw-model methods
  return tensors on the model's device while `get_order` ranks host-side,
  `score_ordering` returns CPU tensors, and wrapper `get_explanation` still
  returns numpy arrays.
- **v0.2.8** — multiclass generalisation: `TextRuleOfThumb.get_explanation`
  follows the tabular semantics (signed class-1 `[N, tokens]` for binary,
  full per-class `[N, n_classes, tokens]` for K > 2); `score_ordering`
  defaults to per-step accuracy for any number of classes and accepts
  `return_confusion=True` for per-step K×K confusion counts (rows = true
  label); custom binary-count `metric=` callables are retained.
- **v0.2.7** — tabular `RuleOfThumb.get_explanation` returns signed,
  SHAP-comparable importances: class-1 contributions for binary tasks
  (additive with the class-1 bias), full per-class output for K > 2; unused
  private helpers `_get_exp_abs_sum` / `_get_exp_sum` / `_get_exp_0m1` /
  `_get_exp_1m0` removed.
- **v0.2.6** — the tabular/text explainer wrappers accept `n_classes=` (default
  2, previously hard-coded) and pass it through to the underlying models.
- **v0.2.5** — reproducible seeding: `training_loop` and all three `fit`
  methods accept `seed=` (covering batch shuffling and dropout draws; fits
  seed once up front so the whole fit is one deterministic stream), and the
  tabular/text wrappers thread `seed=` through.
- **v0.2.4** — training hyperparameters are exposed as arguments: `fit` takes
  `pretrain_epochs=` (was hard-coded 5) and `weight_decay=` (was hard-coded
  0.01) on all three variants; `training_loop` takes `swa_burn_in=` (legacy
  default `epochs // 10 + 1`); the tabular/text wrappers thread
  `pretrain_epochs=`, `weight_decay=` and (text) `l1_penalty=` through.
- **v0.2.3** — `get_order` uses a stable descending sort, so units with equal
  importance rank deterministically (earlier index first) instead of
  nondeterministically under quicksort.
- **v0.2.2** — `mins` / `maxs` are now instance attributes set in `RoT.__init__`
  (±inf defaults) instead of class attributes, so unfitted models no longer
  share state across instances.
- **v0.2.1** — reveal curves default to whole units (token per step for text,
  pixel per step for images); `granularity="element"` restores per-element
  curves.
- **v0.2.0** — explicit mask-based padding replaces the implicit `-1`
  sentinel (breaking); ragged text via `pad_sequences`, mixed-size images via
  `pad_images` or per-sample looping.
- **v0.1.x** — initial package port of the original experiment code,
  including dead-code cleanup.
