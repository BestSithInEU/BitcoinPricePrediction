from .base import BaseRegressor
from catboost import CatBoostRegressor


class CatBoostRegressorModel(BaseRegressor):
    def __init__(
        self,
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function="RMSE",
    ):
        self.catboost_model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function=loss_function,
        )
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.model_name = "Cat Boost Regressor Model"

    def fit(self, X, y):
        return self.catboost_model.fit(X, y)

    def predict(self, X):
        return self.catboost_model.predict(X)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "iterations": [100, 200, 500],
            "learning_rate": [0.01, 0.1, 0.2],
            "depth": [4, 6, 10],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
        }
