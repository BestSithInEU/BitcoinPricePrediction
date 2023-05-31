from .base import BaseRegressor
from sklearn.linear_model import ElasticNet


class ElasticNetRegressorModel(BaseRegressor, ElasticNet):
    """
    ElasticNetRegressorModel is a regression model based on the Elastic Net algorithm.

    Parameters:
    ----------
        alpha (float): Constant that multiplies the penalty terms. Default is 1.0.
        l1_ratio (float): The mixing parameter, with 0 <= l1_ratio <= 1. Default is 0.5.

    Methods:
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the ElasticNetRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the ElasticNetRegressorModel.
    """

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
    ):
        ElasticNet.__init__(
            self,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )
        self.name = "ElasticNetRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the ElasticNetRegressorModel using grid search and cross-validation.

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
        param_grid = {
            "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the ElasticNetRegressorModel.

        Parameters:
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns:
        -------
            dict: The current hyperparameters of the ElasticNetRegressorModel.
        """
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
        }
