from .base import BaseRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


class GaussianProcessRegressorModel(BaseRegressor, GaussianProcessRegressor):
    def __init__(
        self,
        kernel=None,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
    ):
        GaussianProcessRegressor.__init__(
            self,
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
        )
        self.name = "GaussianProcessRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "alpha": [1e-10, 1e-8, 1e-6, 1e-4],
            "normalize_y": [True, False],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "alpha": self.alpha,
            "normalize_y": self.normalize_y,
        }
