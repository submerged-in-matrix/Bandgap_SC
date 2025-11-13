from src.modules import *
from features.shap_initial import *

##############################  RF  #####################################
# Best parameters:([('max_depth', 34), ('max_features', 11), ('max_samples', 1.0), ('min_samples_leaf', 1), ('min_samples_split', 2), ('n_estimators', 1000), ('oob_score', False), ('warm_start', False)])
feature_counts_opt = range(27, 133, 1)
results_rf_bayes_opt = {}

# --- Random Forest ---
for k in feature_counts_opt:
    selected_features = shap_series_rf.sort_values(ascending=False).head(k).index
    X_train_k = X_train[selected_features]
    X_test_k = X_test[selected_features]

    # Bayes-tuned RF (for 27 features in the dataset)
    model_rf_bayes = RandomForestRegressor(random_state=42, verbose=0,
                                           n_estimators=1000,
                                           max_depth=34,           # the default value is None, which means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                           max_features=11,        # the default value is 1.0, which means the full dataset is used. 11 is close to the square root of the total features in the original dataset .
                                           min_samples_split=2,    # the default value is 2
                                           min_samples_leaf=1,     # the default value is 1
                                           max_samples=1.0,        # the default value is None, which means the full dataset is used. 1.0 means the full dataset is used.
                                           oob_score= False,       # the default value is False, which means out-of-bag samples are not used for validation.
                                           warm_start= False       # the default value is False, which means the model is not warm-started.
                                           )

    model_rf_bayes.fit(X_train_k, y_train)
    preds_rf_bayes = model_rf_bayes.predict(X_test_k)
    results_rf_bayes_opt[k] = r2_score(y_test, preds_rf_bayes)

top_3_feat_choice_opt = sorted(results_rf_bayes_opt.items(), key=lambda x: x[1], reverse=True)[:3]

##############################  XGB  #####################################

# Best parameters found: ([('colsample_bytree', 0.35638495679812116), ('gamma', 0.0), ('learning_rate', 0.13425429947962628),   
# ('max_depth', 33), ('n_estimators', 1000), ('reg_alpha', 0.3343417337155307), ('reg_lambda', 10.0), ('subsample', 0.9201475922066067)])
feature_counts_xgb = range(27, 133, 1)
results_xgb_bayes = {}
# --- XGBoost ---
for k in feature_counts_xgb:
    selected_features = shap_series_xgb.sort_values(ascending=False).head(k).index
    X_train_k = X_train[selected_features]
    X_test_k = X_test[selected_features]

    # Bayes-tuned XGB (for 73 features in the dataset)
    model_xgb_bayes = XGBRegressor(random_state=42, verbosity=0,
                                    n_estimators=1000,
                                    max_depth=33,                             # the default value is 6 > the maximum depth of a tree is 6.
                                    learning_rate=0.13425429947962628,        # the default value is 0.3 > the learning rate is set to 0.13425429947962628.
                                    subsample=0.9201475922066067,             # the default value is 1.0 > the subsample ratio of the training instances is set to 0.9201475922066067.
                                    colsample_bytree= 0.35638495679812116,    # the default value is 1.0 > the subsample ratio of columns when constructing each tree is set to 0.35638495679812116.
                                    reg_alpha=0.3343417337155307,             # the default value is 0 > the L1 regularization term on weights is set to 0.3343417337155307.
                                    reg_lambda=10.0,                          # the default value is 1 > the L2 regularization term on weights is set to 10.0.
                                    gamma=0.0                                 # the default value is 0 > the minimum loss reduction required to make a further partition on a leaf node is set to 0.0.
                                    )
    model_xgb_bayes.fit(X_train_k, y_train)
    preds_xgb_bayes = model_xgb_bayes.predict(X_test_k)
    results_xgb_bayes[k] = r2_score(y_test, preds_xgb_bayes)

top_3_feat_choice_xgb = sorted(results_xgb_bayes.items(), key=lambda x: x[1], reverse=True)[:3]

##############################  LGBM  #####################################

# #1. When started with Best  found (67) feature sets for not_optimized model:([('colsample_bytree', 0.5), ('learning_rate', 0.02149411536747437), ('max_depth', 33), ('min_child_samples', 20),  
# #('n_estimators', 1000), ('num_leaves', 512), ('reg_alpha', 1e-08), ('reg_lambda', 1e-08), ('subsample', 0.5)])

# feature_counts_lgbm = range(27, 133, 1)
# results_lgbm_bayes= {}

# for k in feature_counts_lgbm:
#     selected_features = shap_series_lgbm.sort_values(ascending=False).head(k).index
#     X_train_k = X_train[selected_features]
#     X_test_k = X_test[selected_features]

#     # Bayes-tuned LGBM # (for 67 features in the dataset)
#     model_lgbm_bayes = LGBMRegressor(random_state=42, verbosity=-1,
#                                      learning_rate=0.02149411536747437,    # the default value is 0.1 > the learning rate is set to 0.02149411536747437.
#                                      num_leaves=512,                       # the default value is 31 > the maximum number of leaves in one tree is set to 512.
#                                      max_depth=33,                         # the default value is -1 (unlimited) > the maximum depth of a tree is set to 33.
#                                      min_child_samples=20,                 # the default value is 20 > the minimum number of data points in a leaf node is set to 20.
#                                      n_estimators=1000,                    # the default value is 100 > the number of boosting iterations is set to 1000.
#                                      subsample=0.5,                        # the default value is 1.0 > the subsample ratio of the training instance is set to 0.5.
#                                      colsample_bytree=0.5,                 # the default value is 1.0 > the subsample ratio of columns when constructing each tree is set to 0.5.
#                                      reg_alpha=1e-08,                      # the default value is 0.0 > the L1 regularization term on weights is set to 1e-08. almost no regularization.
#                                      reg_lambda=1e-08                      # the default value is 0.0 > the L2 regularization term on weights is set to 1e-08. almost no regularization.
#                                      )
    
#     model_lgbm_bayes.fit(X_train_k, y_train)
#     preds_lgbm_bayes = model_lgbm_bayes.predict(X_test_k)
#     results_lgbm_bayes[k] = r2_score(y_test, preds_lgbm_bayes)
    
# top_3_feat_choice_lgbm = sorted(results_lgbm_bayes.items(), key=lambda x: x[1], reverse=True)[:3]
##--------------------------------------------------------------------------------------------------------------------------##

# When started with random and intuitive 50% of initial featureset (66): ([('colsample_bytree', 0.5), ('learning_rate', 0.02805452532134942), ('max_depth', 34), ('min_child_samples', 20),  
#('n_estimators', 945), ('num_leaves', 31), ('reg_alpha', 1e-08), ('reg_lambda', 1e-08), ('subsample', 1.0)])
feature_counts_lgbm_50 = range(27, 133, 1)
results_lgbm_bayes_50= {}

for k in feature_counts_lgbm_50:
    selected_features = shap_series_lgbm.sort_values(ascending=False).head(k).index
    X_train_k = X_train[selected_features]
    X_test_k = X_test[selected_features]

    # Bayes-tuned LGBM # (for 66 features in the dataset)
    model_lgbm_bayes = LGBMRegressor(random_state=42, verbosity=-1,
                                     learning_rate=0.02805452532134942,    # the default value is 0.1 > the learning rate is set to 0.02149411536747437.
                                     num_leaves=31,                        # the default value is 31 > the maximum number of leaves in one tree is set to 512.
                                     max_depth=34,                         # the default value is -1 (unlimited) > the maximum depth of a tree is set to 33.
                                     min_child_samples=20,                 # the default value is 20 > the minimum number of data points in a leaf node is set to 20.
                                     n_estimators=945,                     # the default value is 100 > the number of boosting iterations is set to 1000.
                                     subsample=1.0,                        # the default value is 1.0 > the subsample ratio of the training instance is set to 0.5.
                                     colsample_bytree=0.5,                 # the default value is 1.0 > the subsample ratio of columns when constructing each tree is set to 0.5.
                                     reg_alpha=1e-08,                      # the default value is 0.0 > the L1 regularization term on weights is set to 1e-08. almost no regularization.
                                     reg_lambda=1e-08                      # the default value is 0.0 > the L2 regularization term on weights is set to 1e-08. almost no regularization.
                                     )
    
    model_lgbm_bayes.fit(X_train_k, y_train)
    preds_lgbm_bayes = model_lgbm_bayes.predict(X_test_k)
    results_lgbm_bayes_50[k] = r2_score(y_test, preds_lgbm_bayes)
    
top_3_feat_choice_lgbm_50 = sorted(results_lgbm_bayes_50.items(), key=lambda x: x[1], reverse=True)[:3]