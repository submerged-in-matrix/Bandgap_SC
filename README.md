Band Gap Prediction from Composition-Only Descriptors

This project investigates how far one can push band gap prediction of semiconductors using composition-only descriptors, deliberately excluding structural information.

Motivation

Band gap prediction typically relies on structural fingerprints or DFT-derived quantities. This study isolates the predictive power of purely compositional features to understand the limits of structure-agnostic modeling.

The central question:

How much electronic behavior can be inferred from chemistry alone?

Data

~4,600 semiconductor entries

Source: Matbench dataset (Materials Project origin)

Target: Band gap (continuous regression)

Feature Engineering

Featurization via Matminer

Composition-derived elemental statistics

SHAP-based global importance ranking

Iterative pruning and feature subset optimization

No structural features were used.

Models

Individually optimized tree-based ensemble models:

Random Forest

XGBoost

LightGBM

Hyperparameters were tuned explicitly (not AutoML-driven) to study model behavior under constrained descriptors.

Results

Proxy R² ≈ 0.78

Stable cross-validation performance

Clear feature dominance patterns revealed via SHAP

Observations

Composition carries substantial electronic signal.

Ensemble diversity improves robustness under feature sparsity.

Structural descriptors are likely required for further gains beyond ~0.8 R².

Next Steps

Introduce structure-derived features (e.g., coordination, symmetry proxies)

Incorporate physics-motivated corrections (quantum confinement, SOC proxies)

Expand dataset size and include wider band gap regimes
