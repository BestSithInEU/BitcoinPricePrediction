from .base import BaseRegressor
from sklearn.linear_model import TweedieRegressor


class TweedieRegressorModel(BaseRegressor, TweedieRegressor):
    def __init__(self, power=0, alpha=0.5, link="auto", max_iter=100, tol=0.0001):
        TweedieRegressor.__init__(
            self,
            power=power,
            alpha=alpha,
            link=link,
            max_iter=max_iter,
            tol=tol,
        )
        self.name = "TweedieRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {"power": [0, 1, 2], "alpha": [0.1, 0.5, 1.0]}
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "power": self.power,
            "alpha": self.alpha,
        }
