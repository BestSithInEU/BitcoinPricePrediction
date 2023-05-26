from .base import BaseRegressor
from sklearn.svm import SVR


class SupportVectorRegressorModel(BaseRegressor, SVR):
    def __init__(self, kernel="rbf", degree=3, C=1.0, epsilon=0.1):
        super().__init__(
            kernel=kernel,
            degree=degree,
            C=C,
            epsilon=epsilon,
        )
        self.model_name = "Support Vector Regressor Model"

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "kernel": ["linear", "poly", "rbf"],
            "C": [0.1, 1, 10],
            "epsilon": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "kernel": self.kernel,
            "degree": self.degree,
            "C": self.C,
            "epsilon": self.epsilon,
        }
