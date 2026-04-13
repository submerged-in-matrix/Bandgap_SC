<div align="center">

# 🔬 Band Gap Prediction from Composition Alone

**How far can Magpie descriptors take band gap prediction — without any structural information?**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-100%25-3776AB?logo=python&logoColor=white)](.)
[![Dataset](https://img.shields.io/badge/Dataset-matbench__expt__gap-blue)](.)
[![Best R²](https://img.shields.io/badge/Best_R²-0.775_(LGBM)-brightgreen)](.)

</div>

---

## The Question

Band gap prediction is a well-studied problem in materials informatics, but most high-performing models rely on **structural features** — crystal system, symmetry, coordination environment, density. This project asks a more constrained question: **up to which extent can the band gap of a compound be predicted solely from its chemical composition?**

This matters because structure is not always available. For hypothetical or not-yet-synthesized compounds, only the composition is known. Understanding the ceiling of composition-only models helps calibrate expectations and decide when structural data is truly necessary.

## The Approach: Iterative Feature–Model Co-Optimization

Rather than training one model and reporting a number, this project runs a **four-stage optimization loop** where feature selection and model tuning are interleaved — each stage feeds into the next:

```
Stage 1                    Stage 2                    Stage 3                    Stage 4
─────────────────────     ─────────────────────     ─────────────────────     ─────────────────────
Default models            SHAP feature sweep        BayesSearchCV tuning      SHAP feature sweep
on all 132 features  →    on default models     →   on reduced features  →    on tuned models
                          (find opt. k per model)   (optimize structure)       (re-find opt. k)
                                │                         │                         │
                          RF:  k=27                 RF:  tuned                RF:  k=27
                          XGB: k=73                 XGB: tuned                XGB: k=63
                          LGBM: k=67                LGBM: tuned               LGBM: k=71
```

The key insight: **the optimal feature set size depends on the model's hyperparameters, and the optimal hyperparameters depend on the feature set.** Doing both in a single pass misses the interaction. The iterative approach captures it.

## Dataset & Featurization

**Source:** `matbench_expt_gap` from Matminer — a curated dataset of experimentally measured band gaps.

**Featurization:** Matminer's `ElementProperty` with **Magpie presets** — statistical aggregates (mean, std, min, max, range, mode) over elemental properties (electronegativity, atomic radius, valence electrons, melting point, etc.). This produces **132 composition-only features** per compound. No structural descriptors are used.

## Optimization in Detail

### Stage 1 — Baseline (All Features, Default Models)

Three tree-based regressors are trained on the full 132-feature matrix with default hyperparameters:

| Model | R² | MAE (eV) |
|---|---|---|
| Random Forest | 0.722 | 0.452 |
| XGBoost | 0.753 | 0.442 |
| LightGBM | 0.753 | 0.437 |

### Stage 2 — SHAP Feature Selection (Default Models)

SHAP values (TreeExplainer) are computed independently for each model. Features are ranked by mean absolute SHAP importance, then a sweep from 20 to 132 features identifies the optimal feature count per model:

| Model | Optimal k | R² |
|---|---|---|
| RF | 27 | 0.722 |
| XGB | 73 | 0.753 |
| LGBM | 67 | 0.753 |

A striking finding: **RF achieves its best performance with only 27 features** — less than a quarter of the full set — while the boosting models need 5× more. This reflects fundamental differences in how these model families handle irrelevant features.

### Stage 3 — Bayesian Hyperparameter Tuning

With the reduced feature sets from Stage 2, each model is tuned via **BayesSearchCV** (scikit-optimize) over extensive search spaces:

- **RF:** `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`, `max_samples`, `oob_score`, `warm_start`
- **XGB:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`
- **LGBM:** `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `n_estimators`, `colsample_bytree`, `subsample`, `reg_alpha`, `reg_lambda`

An important methodological detail: **LGBM is tuned twice** — once starting from the Stage 2 optimum (67 features) and once starting from 50% of the full feature set (66 features). The latter path produces a better final model (R² = 0.775), demonstrating that the initial feature set for tuning influences the final optimum.

### Stage 4 — SHAP Re-Optimization on Tuned Models

The tuned models from Stage 3 are fed back through a second SHAP feature sweep. Because the model structure has changed, the optimal feature counts shift:

| Model | Stage 2 k | Stage 4 k | Final R² | Final MAE (eV) |
|---|---|---|---|---|
| RF | 27 | 27 | 0.719 | 0.464 |
| XGB | 73 | 63 | 0.764 | 0.410 |
| **LGBM** | 67 | **71** | **0.775** | **0.397** |

**LGBM emerges as the best model** — gaining features after tuning (67 → 71), not losing them. The tuned model can exploit weak signals that the default model treated as noise.

## Generalizability Check

Model stability is tested by evaluating R² on incrementally growing slices of the test set (5, 10, 15, … samples). This reveals whether the final R² is stable or driven by a few lucky predictions. All three final models show stable convergence, confirming that the performance is robust.

## Feature Importance Shift

SHAP summary plots are generated for both default and optimized versions of each model and compared side-by-side. This reveals how feature importance rankings change after tuning — a useful diagnostic for understanding whether tuning changes *what* the model learns or just *how well* it learns the same patterns. Results are saved in [`some results/`](some%20results/).

## Repository Structure

```
Bandgap_SC/
├── src/                # Core pipeline implementation
├── features/           # Featurization and choosing optimal feature-set
├── data/               # Raw and processed datasets (matbench_expt_gap), data-splitting for ML
├── some results/       # SHAP plots, optimization summaries, comparisons
├── utils/              # Shared utilities
├── .env/               # Environment configuration
├── main.ipynb          # Full experiment notebook (all 4 stages)
└── README.md
```

## Tech Stack

| Component | Technology |
|---|---|
| **Models** | RandomForest, XGBoost, LightGBM |
| **Feature Selection** | SHAP (TreeExplainer, per-model ranking) |
| **Hyperparameter Tuning** | BayesSearchCV (scikit-optimize) |
| **Featurization** | Matminer (`ElementProperty`, Magpie presets) |
| **Dataset** | `matbench_expt_gap` (Matminer benchmark) |
| **Structure Handling** | Pymatgen (Composition objects) |

## Key Takeaways

**Composition-only ceiling:** With Magpie descriptors alone, the best achievable R² is ~0.775 (LGBM). This sets a concrete lower bound for what structural features add — any structure-aware model that does not meaningfully exceed this number is not exploiting structural information effectively.

**Feature–model co-dependence:** The optimal number of features differs dramatically across model families (RF: 27, XGB: 63, LGBM: 71) and shifts after hyperparameter tuning. Selecting features once and applying them to all models — a common shortcut — leaves performance on the table.

**Iterative optimization matters:** The LGBM model's path from R² = 0.732 (full features, default) → 0.753 (SHAP-reduced) → 0.775 (tuned + re-optimized features) shows a 4.3 percentage point gain from the iterative approach, substantially larger than what a single-pass pipeline achieves.

---

<div align="center">

📬 sayeed.shahriar@gmail.com · [GitHub](https://github.com/submerged-in-matrix)

</div>
