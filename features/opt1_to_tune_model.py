from src.modules import *
from data.train_test import X_train, X_test, y_train, y_test
from features.shap_initial import *

# --- Get top features ---
top_features_xgb = shap_series_xgb.sort_values(ascending=False).head(73).index
top_features_lgbm = shap_series_lgbm.sort_values(ascending=False).head(67).index
top_features_rf = shap_series_rf.sort_values(ascending=False).head(27).index

# --- Subset original data while sticking to the original data-split. ---
X_train_xgb = X_train[top_features_xgb]
X_test_xgb = X_test[top_features_xgb]

X_train_lgbm = X_train[top_features_lgbm]
X_test_lgbm = X_test[top_features_lgbm]

X_train_rf = X_train[top_features_rf]
X_test_rf = X_test[top_features_rf] 