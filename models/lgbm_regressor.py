from .base import BaseRegressor
from lightgbm import LGBMRegressor


class LGBMRegressorModel(BaseRegressor, LGBMRegressor):
    """
    LGBMRegressorModel is a regression model based on the LightGBM algorithm.

    Parameters
    ----------
        n_estimators (int): The number of boosting iterations. Default is 100.
        learning_rate (float): The learning rate of the boosting process. Default is 0.1.
        max_depth (int): The maximum depth of each tree. Default is -1 (unlimited).
        num_leaves (int): The maximum number of leaves in each tree. Default is 31.
        random_state (int): The random seed for reproducible results. Default is 95.

    Methods
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the LGBMRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the LGBMRegressorModel.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        num_leaves=31,
        random_state=95,
    ):
        LGBMRegressor.__init__(
            self,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
        )
        self.name = "LightGradientBoostingRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the LGBMRegressorModel using grid search and cross-validation.

        Parameters
        ----------
            X_train (numpy.ndarray): The training features.
            X_val (numpy.ndarray): The validation features.
            y_train (numpy.ndarray): The training target.
            y_val (numpy.ndarray): The validation target.

        Returns
        -------
            tuple: A tuple containing the best estimator and the best parameters found during tuning.
        """
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [-1, 5, 10, 15],
            "num_leaves": [20, 31, 40, 50],
            "random_state": [self.random_state],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the LGBMRegressorModel.

        Parameters
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict: The current hyperparameters of the LGBMRegressorModel.
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "random_state": self.random_state,
        }
