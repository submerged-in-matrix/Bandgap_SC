# Band Gap Prediction from Composition-Only Descriptors

## Overview

This project investigates how far band gap prediction of semiconductors can be pushed using **composition-only descriptors**, deliberately excluding structural information.

The central question I wondered:

> How much electronic behavior can be inferred from chemistry alone?

---

## Dataset

- ~4,600 semiconductor entries  
- Source: Matbench (Materials Project origin)  
- Target: Continuous band gap regression  

---

## Feature Engineering

- Featurization via **Matminer, Pymatgen**
- Elemental statistical descriptors
- SHAP-based global importance ranking
- Iterative feature pruning

No structural features were used.

---

## Models

Individually optimized ensemble regressors:

- Random Forest  
- XGBoost  
- LightGBM  

Hyperparameters were manually tuned to analyze model behavior under descriptor constraints.

---

## Results

- Proxy R² ≈ **0.78**
- Stable cross-validation performance
- Clear feature dominance patterns from SHAP analysis

---

## Observations

Composition carries substantial electronic signal, but structural descriptors (e.g., Smooth Overlap of Atomic Positions (SOAP)) are likely required to exceed ~0.8 R². 

---

## Future Work

- Incorporate structural features  
- Expand dataset scale
- Expand to various materials systems other than Semiconductors.
