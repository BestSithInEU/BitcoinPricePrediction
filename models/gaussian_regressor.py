from .base import BaseRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


class GaussianProcessRegressorModel(BaseRegressor, GaussianProcessRegressor):
    """
    GaussianProcessRegressorModel is a regression model based on the Gaussian Process algorithm.

    Parameters:
    ----------
        kernel (kernel object): The kernel specifying the covariance function of the Gaussian process. Default is None.
        alpha (float): Value added to the diagonal of the kernel matrix during fitting. Default is 1e-10.
        optimizer (string or callable): The optimizer to use for optimizing the kernel's parameters. Default is "fmin_l_bfgs_b".
        n_restarts_optimizer (int): The number of restarts of the optimizer for optimizing the kernel's parameters. Default is 0.
        normalize_y (bool): Whether to normalize the target values. Default is False.
        copy_X_train (bool): Whether to make a copy of the training data. Default is True.

    Methods:
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the GaussianProcessRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the GaussianProcessRegressorModel.
    """

    def __init__(
        self,
        kernel=None,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
    ):
        GaussianProcessRegressor.__init__(
            self,
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
        )
        self.name = "GaussianProcessRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the GaussianProcessRegressorModel using grid search and cross-validation.

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
            "alpha": [1e-10, 1e-8, 1e-6, 1e-4],
            "normalize_y": [True, False],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the GaussianProcessRegressorModel.

        Parameters:
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns:
        -------
            dict: The current hyperparameters of the GaussianProcessRegressorModel.
        """

        return {
            "alpha": self.alpha,
            "normalize_y": self.normalize_y,
        }
