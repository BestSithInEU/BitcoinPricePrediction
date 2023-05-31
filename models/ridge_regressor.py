from .base import BaseRegressor
from sklearn.linear_model import Ridge


class RidgeRegressorModel(BaseRegressor, Ridge):
    """
    RidgeRegressorModel is a regression model based on the Ridge regression algorithm.

    Parameters:
    ----------
        alpha (float): The regularization parameter. Default is 1.0.

    Methods:
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the RidgeRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the RidgeRegressorModel.
    """

    def __init__(
        self,
        alpha=1.0,
    ):
        Ridge.__init__(
            self,
            alpha=alpha,
        )
        self.name = "RidgeRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the RidgeRegressorModel using grid search and cross-validation.

        Parameters:
        ----------
            X_train (numpy.ndarray): The training features.
            X_val (numpy.ndarray): The validation features.
            y_train (numpy.ndarray): The training target.
            y_val (numpy.ndarray): The validation target.

        Returns:
        -------
            tuple: A tuple containing the best estimator and the best parameters found during tuning.
        """
        param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the RidgeRegressorModel.

        Parameters:
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns:
        -------
            dict: The current hyperparameters of the RidgeRegressorModel.
        """
        return {
            "alpha": self.alpha,
        }
