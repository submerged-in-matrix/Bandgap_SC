# --- Imports ---   
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import shap

from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.base import MultipleFeaturizer
from pymatgen.core import Composition

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

from skopt import BayesSearchCV
import optuna
from optuna.integration import XGBoostPruningCallback
from optuna.integration import LightGBMPruningCallback
import joblib

from skopt.space import Integer, Categorical, Real
from sklearn.model_selection import cross_val_score, KFold