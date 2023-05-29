from .base import BaseRegressor
from sklearn.linear_model import PassiveAggressiveRegressor


class PassiveAggressiveRegressorModel(BaseRegressor, PassiveAggressiveRegressor):
    def __init__(
        self,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        verbose=0,
        loss="epsilon_insensitive",
        epsilon=0.1,
        random_state=None,
        warm_start=False,
        average=False,
    ):
        PassiveAggressiveRegressor.__init__(
            self,
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            loss=loss,
            epsilon=epsilon,
            random_state=random_state,
            warm_start=warm_start,
            average=average,
        )
        self.name = "PassiveAggressiveRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        param_grid = {
            "C": [0.5, 1.0, 1.5],
            "max_iter": [500, 1000, 1500],
            "tol": [1e-3, 1e-4, 1e-2],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "shuffle": self.shuffle,
            "verbose": self.verbose,
            "loss": self.loss,
            "epsilon": self.epsilon,
            "random_state": self.random_state,
            "warm_start": self.warm_start,
            "average": self.average,
        }
