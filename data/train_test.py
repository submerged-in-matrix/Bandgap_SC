from data.load_featurized import X, y
from src.modules import *
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)