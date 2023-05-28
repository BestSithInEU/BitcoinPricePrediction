from .base import BaseRegressor
from xgboost import XGBRegressor


class XGBRegressorModel(BaseRegressor, XGBRegressor):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=1,
        colsample_bytree=1,
    ):
        XGBRegressor.__init__(
            self,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )
        self.name = "eXtremeGradientBoostingRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10, 15],
            "min_child_weight": [1, 2, 5],
            "gamma": [0, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }
