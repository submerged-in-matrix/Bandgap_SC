from src.models.predict_models import *
from src.modules import *
from data.train_test import *  

# # Evaluate & Report
r2_RF_Final = r2_score(y_test,  y_pred_RF_Final)
print("R²_RF (final model):", r2_RF_Final)

mae_RF_Final = mean_absolute_error(y_test, y_pred_RF_Final)
print("MAE_RF (final model):", mae_RF_Final)

# # Evaluate & Report
r2_XGB_Final = r2_score(y_test,  y_pred_XGB_Final)
print("R²_XGB (final model):", r2_XGB_Final)

mae_XGB_Final = mean_absolute_error(y_test, y_pred_XGB_Final)
print("MAE_XGB (final model):", mae_XGB_Final)

# # Evaluate & Report
r2_LGBM_Final = r2_score(y_test,  y_pred_LGBM_Final)
print("R²_LGBM (final model):", r2_LGBM_Final)

mae_LGBM_Final = mean_absolute_error(y_test, y_pred_LGBM_Final)
print("MAE_LGBM (final model):", mae_LGBM_Final)