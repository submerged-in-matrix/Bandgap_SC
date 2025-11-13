from src.modules import *
from data.train_test import X_train, X_test, y_train, y_test
from features.shap_initial import *

feature_counts_primary = range(20, 132, 1)
results_rf_bayes_primary = {}

# --- Random Forest ---
for k in feature_counts_primary:
    selected_features = shap_series_rf.sort_values(ascending=False).head(k).index
    X_train_k = X_train[selected_features]
    X_test_k = X_test[selected_features]

    # Bayes-tuned RF
    model_rf_bayes = RandomForestRegressor(random_state=42, verbose= 0)
    model_rf_bayes.fit(X_train_k, y_train)
    preds_rf_bayes = model_rf_bayes.predict(X_test_k)
    results_rf_bayes_primary[k] = r2_score(y_test, preds_rf_bayes)

top_3_feat_choice_primary = sorted(results_rf_bayes_primary.items(), key=lambda x: x[1], reverse=True)[:3]

print("🔝 Top 3 feature counts by R² score:")
for rank, (k_val, r2_val) in enumerate(top_3_feat_choice_primary, start=1):
    print(f"{rank}. k = {k_val}, R² = {r2_val:.4f}")
   
# plt.figure(figsize=(10, 6))
# plt.plot(feature_counts_primary, list(results_rf_bayes_primary.values())  , label="RF - Bayes", marker='o')
# plt.xlabel("Number of Features (k) for non-optimized RF")
# plt.title("R² Score vs Number of Features (k) for non-optimized Random Forest")
# plt.ylabel("R² Score")

results_xgb_bayes_primary = {}
# --- XGBoost ---
for k in feature_counts_primary:
    selected_features = shap_series_xgb.sort_values(ascending=False).head(k).index
    X_train_k = X_train[selected_features]
    X_test_k = X_test[selected_features]

    # Bayes-tuned XGB
    model_xgb_bayes = XGBRegressor(random_state=42, verbosity=0)
    model_xgb_bayes.fit(X_train_k, y_train)
    preds_xgb_bayes = model_xgb_bayes.predict(X_test_k)
    results_xgb_bayes_primary[k] = r2_score(y_test, preds_xgb_bayes)

top_3_feat_choice_xgb_primary = sorted(results_xgb_bayes_primary.items(), key=lambda x: x[1], reverse=True)[:3]
print("🔝 Top 3 feature counts by R² score for XGBoost:")
for rank, (k_val, r2_val) in enumerate(top_3_feat_choice_xgb_primary, start=1):
    print(f"{rank}. k = {k_val}, R² = {r2_val:.4f}")

# plt.figure(figsize=(10, 6))
# plt.plot(feature_counts_primary, list(results_xgb_bayes_primary.values()), label="XGB - Bayes", marker='o')
# plt.xlabel("Number of Features (k) for non-optimized XGBoost")
# plt.title("R² Score vs Number of Features (k) for non-optimized XGBoost")
# plt.ylabel("R² Score") 

results_lgbm_bayes_primary = {}

for k in feature_counts_primary:
    selected_features = shap_series_lgbm.sort_values(ascending=False).head(k).index
    X_train_k = X_train[selected_features]
    X_test_k = X_test[selected_features]

    # Bayes-tuned LGBM
    model_lgbm_bayes = LGBMRegressor(random_state=42, verbosity=-1)
    
    model_lgbm_bayes.fit(X_train_k, y_train)
    preds_lgbm_bayes = model_lgbm_bayes.predict(X_test_k)
    results_lgbm_bayes_primary[k] = r2_score(y_test, preds_lgbm_bayes)
    
top_3_feat_choice_lgbm_primary = sorted(results_lgbm_bayes_primary.items(), key=lambda x: x[1], reverse=True)[:3]
print("🔝 Top 3 feature counts by R² score for LightGBM:")
for rank, (k_val, r2_val) in enumerate(top_3_feat_choice_lgbm_primary, start=1):
    print(f"{rank}. k = {k_val}, R² = {r2_val:.4f}")
    
# plt.figure(figsize=(10, 6))
# plt.plot(feature_counts_primary, list(results_lgbm_bayes_primary.values()), label="LGBM - Bayes", marker='o')
# plt.xlabel("Number of Features (k) for non-optimized LightGBM")
# plt.ylabel("R² Score")
# plt.title("R² Score vs Number of Features (k) for non-optimized LightGBM")