from .base import BaseRegressor
from sklearn.linear_model import BayesianRidge


class BayesianRidgeRegressorModel(BaseRegressor, BayesianRidge):
    """
    BayesianRidgeRegressorModel is a regression model that performs Bayesian ridge regression.

    Parameters
    ----------
        n_iter : int
            The maximum number of iterations. Default is 300.
        tol : float
            The tolerance for stopping criteria. Default is 0.001.
        alpha_1 : float
            Hyperparameter for the Gamma distribution prior over the alpha parameter. Default is 1e-06.
        alpha_2 : float
            Hyperparameter for the Gamma distribution prior over the alpha parameter. Default is 1e-06.
        lambda_1 : float
            Hyperparameter for the Gamma distribution prior over the lambda parameter. Default is 1e-06.
        lambda_2 : float
            Hyperparameter for the Gamma distribution prior over the lambda parameter. Default is 1e-06.
    """

    def __init__(
        self,
        n_iter=300,
        tol=0.001,
        alpha_1=1e-06,
        alpha_2=1e-06,
        lambda_1=1e-06,
        lambda_2=1e-06,
    ):
        BayesianRidge.__init__(
            self,
            n_iter=n_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )
        self.name = "BayesianRidgeRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the BayesianRidgeRegressorModel using grid search and cross-validation.

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

        param_grid = {
            "n_iter": [100, 200, 300, 400],
            "alpha_1": [1e-06, 1e-05, 1e-04],
            "alpha_2": [1e-06, 1e-05, 1e-04],
            "lambda_1": [1e-06, 1e-05, 1e-04],
            "lambda_2": [1e-06, 1e-05, 1e-04],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the BayesianRidgeRegressorModel.

        Parameters
        ----------
            deep : bool
                If True, return the parameters of all sub-objects that are estimators. If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict
                The current hyperparameters of the BayesianRidgeRegressorModel.
        """

        return {
            "n_iter": self.n_iter,
            "alpha_1": self.alpha_1,
            "alpha_2": self.alpha_2,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
        }
