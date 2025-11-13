import warnings
warnings.filterwarnings("ignore")
from src.modules import *
from features.opt1_to_tune_model import *

# Define the search space including defaults and categorical params
main_search_space_lgbm = {
    
    "learning_rate": Real(0.01, 0.5, prior="log-uniform"),      # default: 0.1
    "num_leaves": Integer(31, 512),                             # default: 31, but we allow larger values 
    "max_depth": Categorical([-1] + list(range(30, 51))),       # default: -1 (unlimited), but we allow 30-50 for exploration
    "min_child_samples": Integer(20, 100),                      # default: 20, but we explore some higher values
    "n_estimators": Integer(100, 1000),                         # default: 100
    "subsample": Real(0.5, 1.0),                                # default: 1.0 (if boosting='gbdt')
    "colsample_bytree": Real(0.5, 1.0),                         # default: 1.0
    "reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),         # default: 0.0, but we allow larger values for exploration
    "reg_lambda": Real(1e-8, 10.0, prior="log-uniform")         # default: 0.0, but we allow larger values for exploration
 }

# Step 1: Wrap default configuration as single-point Categorical search space
default_params_lgbm = {
    # "learning_rate": Categorical([0.1]),
    # "num_leaves": Categorical([31]),
    # "max_depth": Categorical([-1]),
    # "min_child_samples": Categorical([20]),
    # "n_estimators": Categorical([100]),
    # "subsample": Categorical([1.0]),
    # "colsample_bytree": Categorical([1.0]),
    # "reg_alpha": Categorical([0.0]),
    # "reg_lambda": Categorical([0.0]),
    "random_state": Categorical([42]),
    "verbosity": Categorical([-1])    
    
}
# Step 2: Compose search_spaces as a list of (space, n_iter) pairs
search_spaces = [
    (default_params_lgbm, 1),                # Force 1 evaluation of the default model
    (main_search_space_lgbm, 99)             # Remaining search iterations
]

# Bayesian optimization
opt_lgbm = BayesSearchCV(
    estimator=LGBMRegressor(random_state=42, verbosity=-1),
    search_spaces=search_spaces,
    n_iter=100,  # deeper exploration
    scoring="r2",
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose= 0
)

# Fit with training data
opt_lgbm.fit(X_train_lgbm, y_train)

# Predict and evaluate
y_pred_bayes_lgbm = opt_lgbm.predict(X_test_lgbm)
r2_bayes_lgbm = r2_score(y_test, y_pred_bayes_lgbm)
mae_bayes_lgbm = mean_absolute_error(y_test, y_pred_bayes_lgbm)

print(f"Bayes optimized LGBM R²: {r2_bayes_lgbm:.4f}")
print("MAE for LGBM after Bayesian optimization:", mae_bayes_lgbm)
print("Best parameters found:", opt_lgbm.best_params_)