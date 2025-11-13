from src.modules import *
from features.opt1_to_tune_model import *
import warnings
warnings.filterwarnings("ignore")

main_search_space_rf = {
    'n_estimators': Integer(100, 1000),                                      # default: 100
    'max_depth': Categorical([None] + list(range(10, 101))),                 # default: None, explore 10–50
    'max_features': Categorical(['sqrt', 'log2'] + list(range(8, 28))),      # default: 1.0 (interpreted), allow categorical + int
    'min_samples_split': Integer(2, 50),                                     # default: 2
    'min_samples_leaf': Integer(1.0, 10),                                    # default: 1
    'max_samples': Real(0.1, 1.0),                                           # default (None → full dataset)
    'oob_score': Categorical([False, True]),                                 # default: False
    'warm_start': Categorical([False, True])                                 # default: False
}

# Define default parameter configuration to force evaluation
default_params_rf ={
    # 'n_estimators': Categorical([100]),
    # 'max_depth': Categorical([None]),
    # 'max_features': Categorical([27]),
    # 'min_samples_split': Categorical([2]),
    # 'min_samples_leaf': Categorical([1]),
    # 'max_samples': Categorical([1.0]),
    # 'oob_score': Categorical([False]),
    # 'warm_start': Categorical([False]),
    'verbose' : Categorical([0]),  # Set verbose to 0 for no output during fitting
    'random_state': Categorical([42])  # Ensure reproducibility
} 

# Compose search_spaces as a list of (space, n_iter) pairs
search_space_rf = [
    (default_params_rf, 1),        # Force 1 evaluation of the default config
    (main_search_space_rf, 99) ]   # Remaining search iterations

# Create the BayesSearchCV object
opt_rf = BayesSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    search_spaces=search_space_rf,
    n_iter=100,  # number of iterations, i.e., number of different parameter combinations to try
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

# Fit to the reduced feature training data
opt_rf.fit(X_train_rf, y_train)  

# Predict and evaluate on the test set
y_pred_red_rf = opt_rf.predict(X_test_rf)
r2_bayes_red_rf = r2_score(y_test, y_pred_red_rf)
mae_bayes_red_rf = mean_absolute_error(y_test, y_pred_red_rf)


print("R² after Bayesian optimization with reduced features training set:", r2_bayes_red_rf)
print("MAE after Bayesian optimization:", mae_bayes_red_rf)
print("Best parameters found:", opt_rf.best_params_)