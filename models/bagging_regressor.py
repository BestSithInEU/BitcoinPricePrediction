from .base import BaseRegressor
from sklearn.ensemble import BaggingRegressor


class BaggingRegressorModel(BaseRegressor, BaggingRegressor):
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        random_state=95,
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.model_name = "BaggingRegressorModel"

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "n_estimators": [50, 100, 200],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
