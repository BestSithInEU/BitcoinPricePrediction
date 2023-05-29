from .base import BaseRegressor
from sklearn.linear_model import BayesianRidge


class BayesianRidgeRegressorModel(BaseRegressor, BayesianRidge):
    def __init__(
        self,
        n_iter=300,
        tol=0.001,
        alpha_1=1e-06,
        alpha_2=1e-06,
        lambda_1=1e-06,
        lambda_2=1e-06,
    ):
        BayesianRidge.__init__(
            self,
            n_iter=n_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )
        self.name = "BayesianRidgeRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "n_iter": [100, 200, 300, 400],
            "alpha_1": [1e-06, 1e-05, 1e-04],
            "alpha_2": [1e-06, 1e-05, 1e-04],
            "lambda_1": [1e-06, 1e-05, 1e-04],
            "lambda_2": [1e-06, 1e-05, 1e-04],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "n_iter": self.n_iter,
            "alpha_1": self.alpha_1,
            "alpha_2": self.alpha_2,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
        }
