from .base import BaseRegressor
from sklearn.svm import NuSVR


class NuSVRRegressorModel(BaseRegressor, NuSVR):
    def __init__(
        self,
        nu=0.5,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        tol=0.001,
        cache_size=200,
    ):
        NuSVR.__init__(
            self,
            nu=nu,
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            tol=tol,
            cache_size=cache_size,
        )
        self.name = "NuSVRRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "nu": [0.25, 0.5, 0.75],
            "C": [0.1, 1, 10],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "nu": self.nu,
            "C": self.C,
            "kernel": self.kernel,
        }
