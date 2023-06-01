from .base import BaseRegressor
from sklearn.linear_model import Lasso


class LassoRegressorModel(BaseRegressor, Lasso):
    """
    LassoRegressorModel is a regression model based on the Lasso (L1 regularization) algorithm.

    Parameters
    ----------
        alpha : float
            The regularization strength. Default is 1.0.
    """

    def __init__(
        self,
        alpha=1.0,
    ):
        Lasso.__init__(
            self,
            alpha=alpha,
        )
        self.name = "LassoRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the LassoRegressorModel using grid search and cross-validation.

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
            tuple: A tuple containing the best estimator and the best parameters found during tuning.
        """
        param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the LassoRegressorModel.

        Parameters
        ----------
            deep : bool
                If True, return the parameters of all sub-objects that are estimators. If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict
                The current hyperparameters of the LassoRegressorModel.
        """
        return {
            "alpha": self.alpha,
        }
