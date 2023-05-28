from .base import BaseRegressor
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeRegressorModel(BaseRegressor, DecisionTreeRegressor):
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
    ):
        DecisionTreeRegressor.__init__(
            self,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        self.name = "DecisionTreeRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "max_depth": [5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", None],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
        }
