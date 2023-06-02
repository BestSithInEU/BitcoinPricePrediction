from .base import BaseRegressor
from catboost import CatBoostRegressor


class CatBoostRegressorModel(BaseRegressor, CatBoostRegressor):
    """
    CatBoostRegressorModel is a gradient boosting regression model that uses the CatBoost algorithm.

    Parameters
    ----------
        iterations : int
            The number of boosting iterations. Default is 500.
        learning_rate : float
            The learning rate for boosting. Default is 0.1.
        depth : int
            The depth of the trees. Default is 6.
        loss_function : str
            The loss function to optimize. Default is 'RMSE'.
    """

    def __init__(
        self,
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function="RMSE",
    ):
        CatBoostRegressor.__init__(
            self,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function=loss_function,
        )
        self.name = "CatBoostRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the CatBoostRegressorModel using grid search and cross-validation.

        Parameters
        ----------
            X_train : numpy.ndarray
                The training features.
            X_val : numpy.ndarray
                The validation features.
            y_train : numpy.ndarray
                The training target.
            y_val : numpy.ndarray
                The validation target.

        Returns
        -------
            tuple
                A tuple containing the best estimator and the best parameters found during tuning.
        """
        param_grid = {
            "iterations": [100, 200, 500],
            "learning_rate": [0.01, 0.1, 0.2],
            "depth": [4, 6, 10],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the CatBoostRegressorModel.

        Parameters
        ----------
            deep : bool
                If True, return the parameters of all sub-objects that are estimators. If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict
                The current hyperparameters of the CatBoostRegressorModel.
        """
        return {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
        }
