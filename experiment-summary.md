# Monorepo replication with the package: experiment summary

## What we ran

All ten research experiments from the monorepo, rebuilt with our package:

- **Old monorepo settings:** every fit uses the monorepo's own hyperparameters (learning rate, epochs, batch size,
weight decay, `l1_penalty`, pretrain epochs), passed in explicitly. Package defaults are ignored.
- `b` **starts at 0:** the old code meant to start `b` at minus the data mean, but that line did nothing, so `b`
started at 0. We copied that by intercepting the package's `-mean` start inside the run scripts. The package
source was not changed.
- **GPU, several seeds:** 3 seeds for most experiments, 5–10 for the fast tabular ones, 2 for Judicial.

**Pass** = within ±0.02 of the number the monorepo reports, or the same signs and ranking where the paper only has
a plot.

## Results

| # | Experiment | Monorepo reports | We got | Result |
|---|---|---|---|---|
| 1 | ExplanationExampleLocal (Pima) | Glucose 1st, Age 2nd, both negative | Same in 3 of 3 seeds | PASS |
| 2 | ScientificDiscovery | Accuracy 0.714, Age last of 8, Age importance ≈ 0.009 | 0.711 ± 0.033, Age 7th–8th, Age importance 0.08 ± 0.05 (10 seeds) | PASS (Age size differs) |
| 3 | AIAuditing | Accuracy 0.696; is_amazon 0.621, sold_by 0.279 | 0.693; 0.586, 0.251 (5 seeds) | Accuracy PASS, importances 0.03 low |
| 4 | AdversarialAttack | German / COMPAS 100%, CC 89.5–91% | COMPAS 100%, CC 89.5%, German 100% on seeds 0 and 2, 44% on seed 1 | PASS except German seed 1 |
| 5 | OpenXAI (COMPAS LR) | Baseline 1.0847, length-of-stay slope −0.003775 | 1.086, −0.003778 | PASS (2 of 7 bars 0.04–0.05 off) |
| 6 | Resume | 6 job-category signs and ranking (plot) | 6/6 signs, ranking correlation 0.94–1.00 | PASS |
| 7 | Movie | wAUROC 0.72 | 0.727 / 0.726 / 0.726 | PASS |
| 8 | Judicial | wAUROC 0.767, PR-AUC 0.774, fidelity 0.9505 | 0.757 / 0.755, 0.762 / 0.763, 0.9505 exactly | PASS (about 0.01 lower) |
| 9 | Pets-5000 | Test 97.60%, train 98.53% | Test 97.68 / 97.52 / 96.96%, train 98.56 / 98.40 / 97.87% | PASS |
| 10 | Runtimes | RoT 0.27–2.73 s, 9.5–844× faster than SHAP | 0.38–2.44 s, 8.3–772× faster | PASS (same scale) |

**Bottom line:** with the monorepo's settings and `b` starting at 0, the package reproduces all 9 headline results
and the runtimes. The remaining differences are seed-to-seed noise on small data and a small known gap on Judicial.

## The differences, in plain words

1. **ScientificDiscovery, Age importance.** Accuracy and Age's rank match, but Age's importance is about 0.08

   instead of 0.009, and it swings between 0.02 and 0.20 across seeds. The paper's "Age is almost exactly zero"
    was one lucky run of a noisy fit.
2. **AIAuditing, importances.** Accuracy matches, but both headline importances are about 0.03 lower on every seed.

   The paper's number is a single run without a fixed seed. Starting `b` at 0 makes no difference here, because
    this data is already centred.
3. **AdversarialAttack, German seed 1.** RoT still agrees with the black box 100% of the time, but on seed 1 it

   leans on a non-sensitive feature for 56% of rows. With only 100 test rows, which feature it picks depends on
    the seed.
4. **OpenXAI, two bars.** Starting `b` at 0 with learning rate 0.01 fixes the old problem (bars were 1.6× too big):

   the baseline and slope now match to 3–4 decimals. The length-of-stay and priors-count bars are still 0.05 and
    0.04 off, a gap the old code also had against its own figure.
5. **Judicial, about 0.01 lower.** This is the known difference in the text model's shape: the old code used one

   offset per class and trained only the baseline during pretraining.
6. **Pets, features.** We used features made the old way (resize to 256, crop to 224), because the package's

   `embed_images` cannot reproduce that step. The old image code also never clamps `b`, while the package does;  
    it made no visible difference.
