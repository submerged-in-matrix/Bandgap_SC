from src.modules import *
from data.train_test import X_train, y_train, X_test, y_test

# --- Train Model ---
model_rf = RandomForestRegressor(random_state=42,
                                 verbose= 0)
model_rf.fit(X_train, y_train)

model_lgbm = LGBMRegressor(random_state=42,
                          verbosity = 0)
model_lgbm.fit(X_train, y_train)

model_xgb = XGBRegressor(random_state=42,
                         verbosity =0)
model_xgb.fit(X_train, y_train)

# --- Evaluate ---
y_pred_rf = model_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"RF MAE: {mae_rf:.4f}")
print(f"RF R² Score: {r2_rf:.4f}")

y_pred_xgb = model_xgb.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGB MAE: {mae_xgb:.3f}")
print(f"XGB R² Score: {r2_xgb:.5f}")

y_pred_lgbm = model_lgbm.predict(X_test)
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)
print(f"LGBM MAE: {mae_lgbm:.3f}") 
print(f"LGBM R² Score: {r2_lgbm:.5f}")