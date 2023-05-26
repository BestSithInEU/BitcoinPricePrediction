from .base import BaseRegressor
from sklearn.linear_model import ElasticNet


class ElasticNetRegressorModel(BaseRegressor, ElasticNet):
    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
    ):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio)
        self.model_name = "Elastic Net Regressor Model"

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
        }
