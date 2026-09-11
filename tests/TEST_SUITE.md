# ruleofthumb — test-suite report

> **Maintenance rule: update this file every time the test suite changes.**
> Adding, removing, renaming, or re-thresholding any test must update the
> exhaustive table in §7 and, where accuracy or timing is affected, §§5–6.
> Accuracy numbers are measured with the exact fit hyperparameters the tests
> use; runtimes are machine-specific baselines (see §1), not guarantees.

## 1. Machine + environment (runtimes baseline)

All numbers below were measured on this machine; yours will vary.

- Hardware: Apple M4 Pro (ARM64, Darwin 24.5.0), 25.7 GB RAM
- Python: 3.10.0 in `.venv` (editable install + `[dev]`)
- Pinned libraries (`requirements.txt`, matches `.venv` exactly):
  `numpy==2.2.6`, `torch==2.13.0`, `torchvision==0.28.0`,
  `transformers==5.15.1`, `matplotlib==3.10.9`, `seaborn==0.13.2`,
  `wordcloud==1.9.6`, `pillow==12.3.0`, `shap==0.49.1`,
  `scikit-learn==1.7.2`, `pandas==2.3.3`, `pytest==9.1.1`, `ruff==0.16.4`
- HF/torchvision weights resolve pinned revisions (`ModernBERT`
  `8949b90`, SST-2 `714eb0f`, MobileNet `IMAGENET1K_V1`) from the local cache;
  no dataset is downloaded at test time. A session-autouse fixture seeds all
  RNGs and single-threads torch, and every integration fit runs on CPU
  (`ROT_TEST_DEVICE` override for local GPU smoke); GPU parity is covered by
  `test_device_parity.py`. Cold run = caches cleared
  (`__pycache__`, `.pytest_cache`); warm run = immediate rerun.

## 2. How to run (all under GNU `timeout`, from the repo root, venv only)

```bash
timeout 3500 .venv/bin/python -m pytest tests -q --durations=25 -p no:cacheprovider
timeout 600 .venv/bin/python -m ruff check .
```

The suite writes only gitignored caches.

## 3. Suite summary (222 tests, all passing 2026-09-09)

| Tier | Files | Tests | Data |
| --- | --- | --- | --- |
| Unit `tests/test_*.py` | 13 files | 160 | Synthetic tensors/arrays, no artifacts |
| Integration `tests/integration/` | 10 files | 60 | Committed artifacts + pinned-revision HF/MobileNet features, RoT fitted live on CPU |

Per-file counts: test_calibrated 4, test_core 18, test_embed 14, test_explain 23, test_faithfulness 3,
test_image 16, test_masks 14, test_nonlinear 15, test_persistence 8, test_plot 19,
test_text 12, test_tune 9, test_vision 5, device_parity 2, gpt_pet 5, image 13, nonlinear 1, persistence 4,
plot 7, tabular 6, tabular_models 11, text 10, tune 3. Total 222.

Standard live-fit hyperparameters (integration RoT fits):
tabular/image `epochs=300, batch_size=5000, learning_rate=0.05, seed=0`;
text `epochs=200, batch_size=500, learning_rate=0.05, seed=0`.

## 4. Artifacts

### 4a. Read at test time (committed under `tests/integration/artifacts/`)

| File | Size | Contents |
| --- | --- | --- |
| `tabular_binary.npz` | 64K | `x` (569,30) scaled breast-cancer features, `y` (569,) LR predictions |
| `breast_cancer_lr.joblib` | 4K | Scaler + LogisticRegression black box |
| `digits_tabular.npz` | 20K | `x` (500,64) flattened digits, `y` (500,) RF predictions |
| `digits_rf.joblib` | 2.3M | RandomForest black box (100 trees) |
| `digits_image.npz` | 92K | `x_multi` (500,1,8,8), `y_multi` (500,) TinyCNN labels, `y_binary` (500,) dense-vs-sparse labels, `x_coords` (500,3,8,8), `y_coords`, `x_bin` (120,1,8,8) |
| `cnn_multiclass.pt` | 28K | TinyCNN 10-class state_dict |
| `cnn_binary.pt` | 12K | TinyCNN binary state_dict |
| `cnn_multiclass_coords.pt` | 28K | TinyCNN 10-class 3-channel state_dict |
| `compas.npz` / `compas_models.joblib` | 8K / 276K | `x` (800,12), `y_gbm/y_svc/y_mlp` (800,) + GBM/SVC/MLP black boxes |
| `compas_preprocessing.joblib` | 4K | Preprocessing bundle (not loaded by tests) |
| `wine.npz` / `wine_models.joblib` | 8K / 456K | `x` (178,13), `y_gbm/y_svc/y_mlp` (178,) + GBM/SVC/MLP black boxes |
| `wine_preprocessing.joblib` | 4K | Preprocessing bundle (not loaded by tests) |
| `reviews.txt` | 4K | Fixed film-review snippets, one per line |
| `pets_labels.csv` | 4K | 20 rows: filename, ground_truth, gpt_label |
| `pet_images/` | 700K | 20 raw cat/dog JPEGs (10/class) |
| `pet_reference_explanations.npz` | 4K | `heatmaps` (20,7,7) regression anchors |
| `manifest.json` | 4K | sha256 + shapes provenance, env versions |

### 4b. Computed live at test time (never committed)

- MobileNetV3-Small `IMAGENET1K_V1` feature maps: pets `(20,576,7,7)`,
  digits-rich `(500,576,7,7)` (torchvision weights from local cache).
- distilbert-SST-2 revision `714eb0f` embeddings/logits for `reviews.txt`
  (HF cache); ModernBERT revision `8949b90` for native-string tests.
- Every RoT explainer under test (only model ever fitted at test time).

### 4c. Written only by `tests/integration/generate_artifacts.py`

All §4a files + `manifest.json` (one-off, fixed seeds, deterministic reruns).
The test suite itself writes nothing except gitignored
`__pycache__/` and `.pytest_cache/`.

## 5. Accuracy report (measured 2026-09-08, same hyperparameters as the tests)

Definitions: **black-box-vs-truth** = black-box model vs ground-truth labels
(where ground truth is available; COMPAS/wine store only black-box outputs by
design, so this is n/a). **RoT-vs-black-box** = surrogate fidelity, the number
the tests assert floors on. **Majority** = always-guess-commonest-class score.

| Case | N | Black-box-vs-truth (measured) | RoT-vs-black-box (measured) | Floor asserted |
| --- | --- | --- | --- | --- |
| tabular breast-cancer/LR | 569 | 0.9877 | 0.9895 (maj 0.633) | `>=0.85` + coeff overlap/corr anchors |
| tabular digits/RF (10-class) | 500 | 1.0000 | 0.9760 (maj 0.100) | `>=0.5` |
| compas gbm / svc / mlp | 800 | n/a (outputs only) | 0.8400 / 0.9187 / 0.8075 | `>=0.75/0.85/0.75` + priors_count top |
| wine gbm / svc / mlp (3-class) | 178 | n/a (outputs only) | 0.9944 ×3 | `>=0.9` ×3 + shared top features |
| image binary dense-vs-sparse | 500 | n/a (median-split labels) | 0.9360 (maj 0.522) | `>=0.9` |
| image 10-class raw C=1 | 500 | n/a (CNN outputs) | 0.1220 (maj 0.104) | `maj..maj+0.05` (must look bad) + top-2 `>=0.75` (measured 0.922; loose: the degenerate landscape amplifies last-ulp BLAS spread) |
| image 10-class coords C=3 | 500 | n/a (CNN outputs) | 0.3240 (maj 0.104) | `>=maj+0.15` (~0.35) and `>=raw+0.15` |
| image 10-class MobileNet rich | 500 | n/a (CNN outputs) | 0.9940 (maj 0.104) | `>=0.8`, `>=maj+0.5`, `>=raw+0.5`, `>=8/10` classes predicted |
| pets GPT-vs-truth / RoT-vs-GPT | 20 | 1.0000 | 1.0000 (floor `>=0.85`; heatmap corr min `>=0.95`, mean `>=0.99`; dog-mass corr `>=0.9`) | as listed |
| text SST-2 (2-class, min conf 0.9953) | reviews | n/a (logits are labels) | 1.0000 | full-curve `>=0.85`, native-string `>=0.8`, sentiment-word hit rates |
| nonlinear wine rbf vs linear | 178 | n/a | shaped 1.0000 vs linear 0.9944 | shaped `>=0.85`, `>=linear-0.02` |
| tune tabular best/refit | 569 | — | 0.9789 / 0.9895 | `>=0.9` / `>=0.9` |
| tune text best/refit | reviews | — | 1.0000 / 1.0000 | `>=0.85` / `>=0.85` |
| tune image best/refit | 500 | — | 0.9520 / 0.9360 | `>=0.9` / `>=0.9` |

Gate note (not committed as a test): TinyCNN-trunk `(500,8,8,8)` maps as a
backbone reached only ~0.49 on 10-class digits, so the committed rich test
uses the MobileNet path (~0.99).

## 6. Runtimes (machine-specific baselines, §1 hardware)

Full suite: **222 passed, warm 254.24s (0:04:14).** The session-autouse
determinism fixture single-threads torch, so this run is slower than the
previous baseline (184 passed, cold 308.81s, warm 149.79s).
Cold≫warm gap is dominated by first-use caches (plot text/wordcloud
163.6s→2.7s) and live feature extraction setups.

Slowest 25, warm run:

| s | Test |
| --- | --- |
| 40.71 | image `test_multiclass_rich_backbone_strong_accuracy` (call) |
| 20.03 | image `test_rich_backbone_feature_shape` (setup: 500× MobileNet forward) |
| 7.40 | text `test_native_string_ingestion_end_to_end` |
| 5.97 | plot `test_text_native_string_pipeline_and_matplotlib_export` |
| 3.87 | image `test_coordinate_channels_restore_multiclass_capacity` |
| 3.43 | tune `test_text_autotune_reaches_sentiment_fidelity` |
| 3.35 | plot `test_word_clouds_render_review_tokens` |
| 3.20 | text `test_native_path_equals_array_path` |
| 2.85 | persistence `test_text_round_trip` (setup) |
| 2.72 | pets `test_feature_maps_shape` (setup: 20× MobileNet forward) |
| 2.71 | plot `test_text_html_highlights_real_tokens` |
| 2.70 | image `test_native_path_equals_array_path` |
| 2.53 | unit tune `test_random_respects_n_candidates_and_space` |
| 2.13 | text `test_seed_reproducibility` |
| 2.03 | text `test_embed_texts_produces_fit_ready_arrays` |
| 1.81 | unit tune `test_search_finds_a_genuinely_good_fit` |
| 1.77 | image `test_binary_seed_reproducibility` |
| 1.74 | tune `test_image_autotune_reaches_binary_fidelity` |
| 1.67 | unit explain `test_fit_auto_detects_modality` |
| 1.64 | tune `test_tabular_autotune_reaches_high_fidelity` |
| 1.49 | pets `test_rot_surrogate_accuracy_against_gpt_labels` |
| 1.46 | image `test_native_image_ingestion_end_to_end` |
| 1.28 | pets `test_dog_images_highlight_the_dog_direction` |
| 1.23 | plot `test_tabular_plots_render_and_export` |
| 1.14 | image `test_multiclass_explanation_shape_and_structure` |

Slowest 5, cold run (rest match warm within ~1s):
163.64 plot `test_text_html_highlights_real_tokens`,
40.82 image rich-backbone strong accuracy,
19.84 rich-backbone feature setup,
7.49 text native-string end-to-end,
6.22 plot text native-string pipeline.

## 7. Exhaustive test table (all 222)

“Pins” = structural/behavioural assertion, no numeric floor. Fit params per §3
unless noted.

### `tests/integration/test_device_parity.py` (2)

| Test | Checks |
| --- | --- |
| `test_cpu_fit_reproduces_exactly` | Same-seed CPU fits agree bit-for-bit |
| `test_accelerator_matches_cpu_within_tolerance` | Available MPS/CUDA agrees with CPU: class agreement `>=0.95`, per-feature corr `>=0.98` (skips on CPU-only runners) |

### `tests/integration/test_gpt_pet_integration.py` (5)

| Test | Checks |
| --- | --- |
| `test_feature_maps_shape` | Live MobileNet maps `(20,576,7,7)`, finite |
| `test_gpt_labels_are_accurate_and_balanced` | Labels {cat,dog}; GPT-vs-truth `>=0.8` (measured 1.0) |
| `test_rot_surrogate_accuracy_against_gpt_labels` | RoT-vs-GPT `>=0.85` (measured 1.0) |
| `test_heatmaps_match_reference_explanations` | Shape (20,7,7); corr min `>=0.95`, mean `>=0.99`; both signs present |
| `test_dog_images_highlight_the_dog_direction` | Mean dog-mass dogs>cats; corr(dog-mass, labels) `>=0.9` |

### `tests/integration/test_image_integration.py` (13)

| Test | Checks |
| --- | --- |
| `test_native_image_ingestion_end_to_end` | JPEG paths→64×64 fit/predict; RoT-vs-green-mass-labels `>=0.9` |
| `test_native_path_equals_array_path` | Path fit ≡ array fit (`allclose`) |
| `test_black_box_labels_match_committed_cnns` | Committed labels ≡ fresh CNN forward pass |
| `test_binary_explanation_shape_and_fidelity` | Shape `(N,H,W)`; accuracy `>=0.9` (measured 0.936); additive w/ bias |
| `test_binary_reveal_curve_recovers_full_accuracy` | Curve endpoint == full accuracy; `curve[-1]>=curve[0]+0.3` |
| `test_binary_seed_reproducibility` | Identical importances across fits |
| `test_multiclass_explanation_shape_and_structure` | Shape `(N,10,H,W)`; accuracy `>=majority` (measured 0.108/maj 0.104) |
| `test_ink_pixels_outrank_empty_borders` | Mean |imp| on ink pixels > borders |
| `test_multiclass_confusion_counts_and_reveal_curve` | Confusion/step counts consistent; final accuracy in `maj..maj+0.05`; top-2 coverage `>=0.75` (measured 0.922; loose: degenerate landscape amplifies last-ulp BLAS spread); curve endpoint == accuracy |
| `test_coordinate_channels_restore_multiclass_capacity` | Accuracy `>=maj+0.15` (measured 0.324); `>=raw+0.15`; reveal/confusion consistent; reproducible |
| `test_rich_backbone_feature_shape` | Live MobileNet digits maps `(500,576,7,7)`; finite; N>49; 10 classes |
| `test_multiclass_rich_backbone_strong_accuracy` | Accuracy `>=0.8` (measured 0.994), `>=maj+0.5`, `>=raw+0.5`; `>=8/10` classes predicted; confusion/reveal consistent |
| `test_default_backbone_path_end_to_end` | Live default backbone paths→maps→fit; 576ch maps; backbone id saved/loaded |

### `tests/integration/test_nonlinear_integration.py` (1)

| Test | Checks |
| --- | --- |
| `test_nonlinear_tabular_matches_or_beats_linear_on_wine` | rbf shaped `>=0.85` and `>=linear-0.02` (measured 1.0 vs 0.9944); per-class shapes |

### `tests/integration/test_persistence_integration.py` (4)

| Test | Checks |
| --- | --- |
| `test_tabular_round_trip` | Save/load identical explanations/predictions/curves (tabular) |
| `test_text_round_trip` | Same for text (numeric arrays) |
| `test_image_round_trip` | Same for image |
| `test_subprocess_load_survives_process_boundary` | `load_explainer` works in a fresh process; committed preds file matches |

### `tests/integration/test_plot_integration.py` (7)

| Test | Checks |
| --- | --- |
| `test_tabular_plots_render_and_export` | SHAP waterfall/force/decision/bar/beeswarm return figures |
| `test_text_html_highlights_real_tokens` | Token HTML colours actual tokens (slow cold: font/cache) |
| `test_saliency_renders_reference_heatmaps_over_pet_images` | Saliency overlays reproduce reference heatmaps |
| `test_word_clouds_render_review_tokens` | Pos/neg/combined clouds render |
| `test_multiclass_tabular_plots_per_class` | Per-class tabular plots |
| `test_image_saliency_from_fitted_explainer` | Saliency from fitted image explainer |
| `test_text_native_string_pipeline_and_matplotlib_export` | String→embed→explain→plot pipeline |

### `tests/integration/test_tabular_integration.py` (6)

| Test | Checks |
| --- | --- |
| `test_binary_explanation_shape_additivity_and_fidelity` | Shape; accuracy `>=0.85` (measured 0.9895; black box 0.9877) |
| `test_binary_top_features_match_logistic_coefficients` | Top-k overlap `>=k//2`; top-7 overlap `>=4`; corr `>=0.5` |
| `test_binary_reveal_curve_recovers_full_accuracy` | Curve shape `(d+1,)`; endpoint == accuracy; `+0.3` lift |
| `test_binary_seed_reproducibility` | Identical importances across fits |
| `test_multiclass_explanation_shape_and_fidelity` | Shape `(N,10,d)`; accuracy `>=0.5` (measured 0.976) |
| `test_multiclass_confusion_counts_match_active_samples` | Per-step confusion counts consistent |

### `tests/integration/test_tabular_models_integration.py` (11)

| Test | Checks |
| --- | --- |
| `test_rot_accuracy_and_explanations_per_black_box[gbm-compas]` | `>=0.75` (measured 0.84); priors_count top |
| `...[svc-compas]` | `>=0.85` (measured 0.9187); priors_count top |
| `...[mlp-compas]` | `>=0.75` (measured 0.8075); priors_count top |
| `...[gbm-wine]` | `>=0.9` (measured 0.9944); shared dominant features |
| `...[svc-wine]` | `>=0.9` (measured 0.9944); shared dominant features |
| `...[mlp-wine]` | `>=0.9` (measured 0.9944); shared dominant features |
| `test_wine_models_share_the_same_dominant_features` | Pairwise top-3 overlap `>=2` |
| `test_reveal_curve_endpoint_equals_full_accuracy[compas]` | Endpoint == accuracy |
| `test_reveal_curve_endpoint_equals_full_accuracy[wine]` | Endpoint == accuracy |
| `test_wine_seed_reproducibility` | Identical importances across fits |
| `test_breast_cancer_fidelity_and_rank_agreement` | Linear/rbf fidelity `>=0.85` (measured 0.965); spearman vs permutation `>0.3` (measured 0.62) |

### `tests/integration/test_text_integration.py` (10)

| Test | Checks |
| --- | --- |
| `test_black_box_predictions_are_confident` | 2 classes; min confidence `>=0.9` (measured 0.9953) |
| `test_explanation_shape_and_padding_zeros` | Shape `(N,tokens)`; pads score exactly 0 |
| `test_sentiment_words_carry_signed_importance` | Pos/neg hit rates `>=0.4` / `>=0.55` |
| `test_top_tokens_carry_sentiment_words` | Top tokens carry sentiment words |
| `test_brilliant_outranks_awful_in_the_pair_review` | “brilliant” > “awful” in pair review (positive overall) |
| `test_reveal_curve_recovers_full_accuracy` | Endpoint == accuracy; full `>=0.85` (measured 1.0) |
| `test_seed_reproducibility` | Identical importances across fits |
| `test_embed_texts_produces_fit_ready_arrays` | `(2,768)` embeddings, masks, tokens, zeroed pads |
| `test_native_string_ingestion_end_to_end` | Raw strings fit; RoT `>=0.8` (measured 1.0) |
| `test_native_path_equals_array_path` | String fit ≡ array fit |

### `tests/integration/test_tune_integration.py` (3)

| Test | Checks |
| --- | --- |
| `test_tabular_autotune_reaches_high_fidelity` | best `>=0.9` (measured 0.9789); refit `>=0.9` (0.9895) |
| `test_text_autotune_reaches_sentiment_fidelity` | best `>=0.85` (1.0); refit `>=0.85` (1.0) |
| `test_image_autotune_reaches_binary_fidelity` | best `>=0.9` (0.9520); refit `>=0.9` (0.9360) |

### `tests/test_core.py` (18)

Pins for base `RoT`: loss decreases + predicts; importance shapes;
ordering pipeline; multiclass/binary `score_ordering` incl. confusion and
metric-conflict error; swapped-points/labels and sample-count guards;
host-side inference tensors;
device resolution/plumbing (cpu); dropout
stochasticity; `mins`/`maxs` instance attrs; stable tie-break; SWA burn-in
arg; fit hyperparameter args; fit + training-loop seeding.

### `tests/test_embed.py` (14)

Pins for `embed_texts`/`TextEmbeddings`: shapes/masks/padding zeros; token
alignment; batch invariance; truncation; tokenizer/model override rules;
device arg; empty/bad-input rejections; `DEFAULT_TEXT_MODEL`; string routing
through facade; strings+explicit-padding rejection; array-fitted string-query
error.

### `tests/test_explain.py` (23)

Pins for `Explainer` facade: modality auto-detect/override; explanation
shapes (tabular/text/image, binary + multiclass); signed outputs; mask-only
text contract; padding-arg/kwarg rejection; out-of-range/empty label
rejection; host-side inference on MPS (skipped without MPS); readonly-array
silence; oversized-batch warning; fit-quality warn/quiet tripwire;
hyperparameter threading; seeding;
device; reveal delegation; `__version__ == "0.2.19"`; removed
`RuleOfThumb`/`TextRuleOfThumb` absent.

### `tests/test_image.py` (16)

Pins for `RoTImage`/paths: importance shapes; fit/score; mask zeroing;
mixed-size batch ≡ per-sample loop; padded fit; `pad_images`; facade binary
pipeline with untrained conv box; padded-batch mask respect; `load_images`
native/fixed/transform paths; warning-free decoding; path routing; path+mask rejection;
path-on-array-fit error.

### `tests/test_masks.py` (14)

Pins for explicit-mask contract + reveal granularity: ragged text; explicit
mask scoring reference; mixed-size image fit/reveal; element-granularity
escape hatches (text/image); truncation defaults; unit≡element at token
boundaries; curve lengths; tabular granularity identity;
legacy imports gone; padding utils exported.

### `tests/test_nonlinear.py` (15)

Pins for `nonlinear=`: default path unchanged; spec validation; identity at
init (rbf/hinge); hyperparameters reach module; manual-formula match;
additive decomposition (K=2/3 × rbf/dict); masked positions stay zero; ring
separation linear cannot do (rbf + hinge); reveal pipeline; persistence
round-trip.

### `tests/test_persistence.py` (8)

Pins for save/load: round-trip identical outputs (tabular/text/image);
`mins`/`maxs` restored; `device=` on load; foreign files rejected;
exact version-match enforcement (mismatch + missing stamp rejected).

### `tests/test_plot.py` (19)

Pins for `rot.plot`: single-row figures (waterfall/force/decision);
batch figures (bar/beeswarm); values/base use class bias; HTML sign colours;
max-tokens truncation; matplotlib text figure; saliency with/without image;
saliency power/trim sweep; multiclass per-class bar/waterfall;
nonpositive power rejected; word clouds figure + single-sign panels.

### `tests/test_calibrated.py` (4)

Pins for calibrated ground-truth fixtures (xai-units lesson, reimplemented):
exact handcrafted linear black box; rank/top-k/sign recovery vs `|w|`;
multiclass per-class top-1; interacting labels trip the quality wire.

### `tests/test_faithfulness.py` (3)

Pins for faithfulness probes on known mechanisms: tabular deletion gap;
localized-square pointing-game; noise columns below signal.

### `tests/test_text.py` (12)

Pins for `RoTText`: importance shapes; mask zeroing; stochastic mask
respect; length normalisation; fit reduces loss; masked fit; `pad_sequences`;
multiclass masked zeros; facade multiclass reveal counts; mask-only
`get_order` contract; tensor≡array mask agreement; seeding.

### `tests/test_tune.py` (9)

Pins for `autotune` on synthetic data: grid enumeration; random
n_candidates/space respect; seeding; split sizes; search beats a bad fit;
refit accurate on all data; multiclass `n_classes` inference + explicit
override; model-kwarg forwarding (`nonlinear`).

### `tests/test_vision.py` (5)

Pins for the image backbone path: stub-trunk map shapes + pooled mask
counts; stub fit end-to-end with `backbone` provenance `None`;
`backbone=None` pixel parity; backbone+transform conflict; save/load
backbone-id round-trip.
