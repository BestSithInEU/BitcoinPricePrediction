from .base import BaseRegressor
from sklearn.ensemble import StackingRegressor


class StackingRegressorModel(BaseRegressor, StackingRegressor):
    def __init__(self, estimators, final_estimator=None, cv=None):
        super().__init__(estimators=estimators, final_estimator=final_estimator, cv=cv)
        self.model_name = "Stacking Regressor Model"

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "final_estimator__C": [0.1, 1.0, 10.0],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "estimators": self.estimators,
            "final_estimator": self.final_estimator,
            "cv": self.cv,
        }
