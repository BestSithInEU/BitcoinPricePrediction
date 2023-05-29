from .base import BaseRegressor
from sklearn.ensemble import ExtraTreesRegressor


class ExtraTreesRegressorModel(BaseRegressor, ExtraTreesRegressor):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
    ):
        ExtraTreesRegressor.__init__(
            self,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        self.name = "ExtraTreesRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }
