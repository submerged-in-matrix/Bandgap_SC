from src.modules import *
from data.train_test import *
from src.models.train_models import *

# ---- XGB Model ----
y_pred_XGB_Final = XGB_FINAL.predict(X_test_XGB_Final)
test_sizes_XGB = []
r2_scores_XGB = []

for size in range(5, len(y_test) + 1, 5):
    y_test_slice = y_test[:size]
    y_pred_slice = y_pred_XGB_Final[:size]
    r2 = r2_score(y_test_slice, y_pred_slice)
    test_sizes_XGB.append(size)
    r2_scores_XGB.append(r2)

# ---- LGBM Model ----
y_pred_LGBM_Final = LGBM_FINAL.predict(X_test_LGBM_Final)
test_sizes_LGBM = []
r2_scores_LGBM = []

for size in range(5, len(y_test) + 1, 5):
    y_test_slice = y_test[:size]
    y_pred_slice = y_pred_LGBM_Final[:size]
    r2 = r2_score(y_test_slice, y_pred_slice)
    test_sizes_LGBM.append(size)
    r2_scores_LGBM.append(r2)

# ---- RF Model ----
y_pred_RF_Final = RF_FINAL.predict(X_test_RF_Final)
test_sizes_RF = []
r2_scores_RF = []

for size in range(5, len(y_test) + 1, 5):
    y_test_slice = y_test[:size]
    y_pred_slice = y_pred_RF_Final[:size]
    r2 = r2_score(y_test_slice, y_pred_slice)
    test_sizes_RF.append(size)
    r2_scores_RF.append(r2)

# ---- Plot Results ----
plt.figure(figsize=(10, 6))
plt.plot(test_sizes_XGB, r2_scores_XGB, marker='o', linestyle='-', color='blue', label='XGB')
plt.plot(test_sizes_LGBM, r2_scores_LGBM, marker='s', linestyle='-', color='green', label='LGBM')
plt.plot(test_sizes_RF, r2_scores_RF, marker='^', linestyle='-', color='red', label='RF')
plt.xlabel('Test Set Size')
plt.ylabel('R² Score')
plt.title('Model Stability: R² vs Test Set Size')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()