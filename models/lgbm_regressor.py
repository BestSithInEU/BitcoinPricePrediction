from .base import BaseRegressor
from lightgbm import LGBMRegressor


class LGBMRegressorModel(BaseRegressor, LGBMRegressor):
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        num_leaves=31,
        random_state=95,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
        )

        self.model_name = "LightGradientBoostingRegressorModel"

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [-1, 5, 10, 15],
            "num_leaves": [20, 31, 40, 50],
            "random_state": [self.random_state],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
        }
