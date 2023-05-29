from .base import BaseRegressor
from sklearn.ensemble import AdaBoostRegressor


class AdaBoostRegressorModel(BaseRegressor, AdaBoostRegressor):
    def __init__(
        self,
        n_estimators=50,
        learning_rate=1.0,
        loss="linear",
    ):
        AdaBoostRegressor.__init__(
            self,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
        )
        self.name = "AdaBoostRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1],
            "loss": ["linear", "square", "exponential"],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
        }
