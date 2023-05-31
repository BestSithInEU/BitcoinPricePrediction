from .base import BaseRegressor
from sklearn.linear_model import TweedieRegressor


class TweedieRegressorModel(BaseRegressor, TweedieRegressor):
    """
    TweedieRegressorModel is a regression model based on the Tweedie regression algorithm.

    Parameters:
    ----------
        power (float): The power parameter in the Tweedie variance function. Default is 0.
        alpha (float): The regularization parameter. Default is 0.5.
        link (str): The link function to use. Default is "auto".
        max_iter (int): The maximum number of iterations. Default is 100.
        tol (float): The tolerance for the optimization algorithm. Default is 0.0001.

    Methods:
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the TweedieRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the TweedieRegressorModel.
    """

    def __init__(self, power=0, alpha=0.5, link="auto", max_iter=100, tol=0.0001):
        TweedieRegressor.__init__(
            self,
            power=power,
            alpha=alpha,
            link=link,
            max_iter=max_iter,
            tol=tol,
        )
        self.name = "TweedieRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the TweedieRegressorModel using grid search and cross-validation.

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
        param_grid = {"power": [0, 1, 2], "alpha": [0.1, 0.5, 1.0]}
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the TweedieRegressorModel.

        Parameters:
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns:
        -------
            dict: The current hyperparameters of the TweedieRegressorModel.
        """
        return {
            "power": self.power,
            "alpha": self.alpha,
        }
