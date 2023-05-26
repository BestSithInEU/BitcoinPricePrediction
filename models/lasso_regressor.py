from .base import BaseRegressor
from sklearn.linear_model import Lasso


class LassoRegressorModel(BaseRegressor, Lasso):
    def __init__(
        self,
        alpha=1.0,
    ):
        super().__init__(alpha=alpha)
        self.model_name = "LassoRegressorModel"

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "alpha": self.alpha,
        }
