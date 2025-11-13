from src.modules import *
from src.models.initial_models import *

explainer_rf= shap.TreeExplainer(model_rf)
shap_values_rf = explainer_rf(X_test)

explainer_xgb = shap.TreeExplainer(model_xgb)
shap_values_xgb = explainer_xgb.shap_values(X_test)

explainer_lgbm = shap.TreeExplainer(model_lgbm)
shap_values_lgbm = explainer_lgbm.shap_values(X_test)

# --- Get mean absolute SHAP values for each feature ---

shap_importance_xgb = np.abs(shap_values_xgb).mean(axis=0)
shap_importance_lgbm = np.abs(shap_values_lgbm).mean(axis=0)

shap_values_rf_array = shap_values_rf.values   # The output -Type- SHAP summary values for Random forest is  # <class 'shap._explanation.Explanation'>. So we need to acces the values attribute of that object.
shap_importance_rf = np.abs(shap_values_rf_array).mean(axis=0)

# --- Create Pandas Series for easy sorting ---
shap_series_xgb = pd.Series(shap_importance_xgb, index=X_test.columns)
shap_series_lgbm = pd.Series(shap_importance_lgbm, index=X_test.columns)
shap_series_rf = pd.Series(shap_importance_rf, index=X_test.columns)