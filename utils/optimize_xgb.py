import warnings
warnings.filterwarnings("ignore")
from src.modules import *
from features.opt1_to_tune_model import *

#Define the search space
main_search_space_xgb = {
    'n_estimators': Integer(100, 1000),                          # default=100
    'max_depth': Integer(6, 50),                                 # default=6
    'learning_rate': Real(0.01, 0.5, prior='uniform'),           # default=0.3
    'subsample': Real(0.5, 1.0),                                 # default=1.0
    'colsample_bytree': Real(0.3, 1.0),                          # default=1.0
    'reg_alpha': Real(1e-8, 10.0, prior='log-uniform'),          # default=0
    'reg_lambda': Real(1e-8, 10.0, prior='log-uniform'),         # default=1
    'gamma': Real(0, 5),                                         # default=0
}

default_params_xgb = {
    'n_estimators': Categorical([100]),
    'max_depth': Categorical([6]),
    'learning_rate': Categorical([0.3]),
    'subsample': Categorical([1.0]),
    'colsample_bytree': Categorical([1.0]),
    'reg_alpha': Categorical([0.0]),
    'reg_lambda': Categorical([1.0]),
    'gamma': Categorical([0.0])
}

search_space_xgb = [
    (default_params_xgb, 1),    # 1 forced evaluation of default config
    (main_search_space_xgb, 99)        # 49 Bayesian iterations
]


opt_xgb = BayesSearchCV (
    
    estimator=XGBRegressor(random_state=42, verbosity=0),
    search_spaces=search_space_xgb,
    n_iter=100,  # Increasing this would lead to more thorough search
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=0,
    random_state=42
    )


# Fit to the reduced feature training data
opt_xgb.fit(X_train_xgb, y_train)

# Predict and evaluate on the test set
y_pred_opt_xgb= opt_xgb.predict(X_test_xgb)
r2_xgb_opt = r2_score(y_test, y_pred_opt_xgb)
mae_xgb_opt = mean_absolute_error(y_test, y_pred_opt_xgb)

print("R² for xgboost after Bayesian optimization:", r2_xgb_opt)
print("MAE for xgboost after Bayesian optimization:", mae_xgb_opt)
print("Best parameters found:", opt_xgb.best_params_)