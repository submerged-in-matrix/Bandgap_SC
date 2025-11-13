from src.modules import *
from data.train_test import *
from features.shap_initial import *

######################################  RF  ######################################
Features_RF_Final = shap_series_rf.sort_values(ascending=False).head(27).index
X_train_RF_Final = X_train[Features_RF_Final]
X_test_RF_Final = X_test[Features_RF_Final]

RF_FINAL = RandomForestRegressor(random_state=42, verbose=0,
                                           n_estimators=1000,
                                           max_depth=34,           # the default value is None, which means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                           max_features=11,        # the default value is 1.0, which means the full dataset is used. 11 is close to the square root of the total features in the original dataset .
                                           min_samples_split=2,    # the default value is 2
                                           min_samples_leaf=1,     # the default value is 1
                                           max_samples=1.0,        # the default value is None, which means the full dataset is used. 1.0 means the full dataset is used.
                                           oob_score= False,       # the default value is False, which means out-of-bag samples are not used for validation.
                                           warm_start= False       # the default value is False, which means the model is not warm-started.
                                           )
# # Train
RF_FINAL.fit(X_train_RF_Final, y_train)

###################################### XGB  ######################################
Features_XGB_Final = shap_series_xgb.sort_values(ascending=False).head(63).index
X_train_XGB_Final = X_train[Features_XGB_Final]
X_test_XGB_Final = X_test[Features_XGB_Final]

XGB_FINAL = XGBRegressor(random_state=42, verbosity=0,
                                    n_estimators=1000,
                                    max_depth=33,                             # the default value is 6 > the maximum depth of a tree is 6.
                                    learning_rate=0.13425429947962628,        # the default value is 0.3 > the learning rate is set to 0.13425429947962628.
                                    subsample=0.9201475922066067,             # the default value is 1.0 > the subsample ratio of the training instances is set to 0.9201475922066067.
                                    colsample_bytree= 0.35638495679812116,    # the default value is 1.0 > the subsample ratio of columns when constructing each tree is set to 0.35638495679812116.
                                    reg_alpha=0.3343417337155307,             # the default value is 0 > the L1 regularization term on weights is set to 0.3343417337155307.
                                    reg_lambda=10.0,                          # the default value is 1 > the L2 regularization term on weights is set to 10.0.
                                    gamma=0.0                                 # the default value is 0 > the minimum loss reduction required to make a further partition on a leaf node is set to 0.0.
                                    )
# # Train
XGB_FINAL.fit(X_train_XGB_Final, y_train)

###################################### LGBM  ######################################
Features_LGBM_Final = shap_series_lgbm.sort_values(ascending=False).head(71).index
X_train_LGBM_Final = X_train[Features_LGBM_Final]
X_test_LGBM_Final = X_test[Features_LGBM_Final]

LGBM_FINAL = LGBMRegressor(random_state=42, verbosity=-1,
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
# # Train
LGBM_FINAL.fit(X_train_LGBM_Final, y_train)