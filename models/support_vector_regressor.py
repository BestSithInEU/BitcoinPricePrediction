from .base import BaseRegressor
from sklearn.svm import SVR


class SupportVectorRegressorModel(BaseRegressor, SVR):
    """
    SupportVectorRegressorModel is a regression model based on the Support Vector Regression algorithm.

    Parameters
    ----------
        kernel : str
            The kernel function to use. Default is "rbf".
        degree : int
            The degree of the polynomial kernel function. Default is 3.
        C : float
            The regularization parameter. Default is 1.0.
        epsilon : float
            The epsilon-tube parameter in the epsilon-insensitive loss function. Default is 0.1.
    """

    def __init__(self, kernel="rbf", degree=3, C=1.0, epsilon=0.1):
        SVR.__init__(
            self,
            kernel=kernel,
            degree=degree,
            C=C,
            epsilon=epsilon,
        )
        self.name = "SupportVectorRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the SupportVectorRegressorModel using grid search and cross-validation.

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
            "kernel": ["linear", "poly", "rbf"],
            "C": [0.1, 1, 10],
            "epsilon": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the SupportVectorRegressorModel.

        Parameters
        ----------
            deep : bool
                If True, return the parameters of all sub-objects that are estimators. If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict
                The current hyperparameters of the SupportVectorRegressorModel.
        """
        return {
            "kernel": self.kernel,
            "degree": self.degree,
            "C": self.C,
            "epsilon": self.epsilon,
        }
