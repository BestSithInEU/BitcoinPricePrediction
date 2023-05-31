from .base import BaseRegressor
from sklearn.svm import NuSVR


class NuSVRRegressorModel(BaseRegressor, NuSVR):
    """
    NuSVRRegressorModel is a regression model based on the Nu-Support Vector Regression algorithm.

    Parameters:
    ----------
        nu (float): An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
                    Default is 0.5.
        C (float): The regularization parameter. Default is 1.0.
        kernel (str): The kernel function to use. Can be "linear", "poly", "rbf", or "sigmoid". Default is "rbf".
        degree (int): The degree of the polynomial kernel function. Default is 3.
        gamma (str or float): The kernel coefficient for "rbf", "poly", and "sigmoid". Can be "scale", "auto", or a float value.
                              Default is "scale".
        coef0 (float): The independent term in the kernel function. Default is 0.0.
        shrinking (bool): Whether to use the shrinking heuristic. Default is True.
        tol (float): The tolerance for stopping criterion. Default is 0.001.
        cache_size (float): The size of the kernel cache in MB. Default is 200.

    Methods:
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the NuSVRRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the NuSVRRegressorModel.
    """

    def __init__(
        self,
        nu=0.5,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        tol=0.001,
        cache_size=200,
    ):
        NuSVR.__init__(
            self,
            nu=nu,
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            tol=tol,
            cache_size=cache_size,
        )
        self.name = "NuSVRRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the NuSVRRegressorModel using grid search and cross-validation.

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
            "nu": [0.25, 0.5, 0.75],
            "C": [0.1, 1, 10],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the NuSVRRegressorModel.

        Parameters:
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns:
        -------
            dict: The current hyperparameters of the NuSVRRegressorModel.
        """
        return {
            "nu": self.nu,
            "C": self.C,
            "kernel": self.kernel,
        }
