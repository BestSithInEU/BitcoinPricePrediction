from .base import BaseRegressor
from sklearn.linear_model import PassiveAggressiveRegressor


class PassiveAggressiveRegressorModel(BaseRegressor, PassiveAggressiveRegressor):
    """
    PassiveAggressiveRegressorModel is a regression model based on the Passive Aggressive algorithm.

    Parameters
    ----------
    C (float): The regularization parameter. Default is 1.0.
    fit_intercept (bool): Whether to calculate the intercept for this model. Default is True.
    max_iter (int): The maximum number of passes over the training data. Default is 1000.
    tol (float): The stopping criterion. Default is 1e-3.
    early_stopping (bool): Whether to use early stopping to terminate training when validation score does not improve. Default is False.
    validation_fraction (float): The proportion of the training data to use as validation set when early stopping is enabled. Default is 0.1.
    n_iter_no_change (int): The maximum number of iterations with no improvement before early stopping. Default is 5.
    shuffle (bool): Whether to shuffle the training data before each epoch. Default is True.
    verbose (int): The verbosity level. Default is 0.
    loss (str): The loss function to use. Can be "epsilon_insensitive" or "squared_epsilon_insensitive".
    Default is "epsilon_insensitive".
    epsilon (float): The epsilon parameter for the epsilon-insensitive loss function. Default is 0.1.
    random_state (int, RandomState instance or None): The seed of the pseudo-random number generator. Default is None.
    warm_start (bool): Whether to reuse the solution of the previous call to fit as initialization. Default is False.
    average (bool): Whether to compute the averaged model. Default is False.

    Methods
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the PassiveAggressiveRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the PassiveAggressiveRegressorModel.
    """

    def __init__(
        self,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        verbose=0,
        loss="epsilon_insensitive",
        epsilon=0.1,
        random_state=None,
        warm_start=False,
        average=False,
    ):
        PassiveAggressiveRegressor.__init__(
            self,
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            loss=loss,
            epsilon=epsilon,
            random_state=random_state,
            warm_start=warm_start,
            average=average,
        )
        self.name = "PassiveAggressiveRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the PassiveAggressiveRegressorModel using grid search and cross-validation.

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
            "C": [0.5, 1.0, 1.5],
            "max_iter": [500, 1000, 1500],
            "tol": [1e-3, 1e-4, 1e-2],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the PassiveAggressiveRegressorModel.

        Parameters
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict: The current hyperparameters of the PassiveAggressiveRegressorModel.
        """
        return {
            "C": self.C,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "shuffle": self.shuffle,
            "verbose": self.verbose,
            "loss": self.loss,
            "epsilon": self.epsilon,
            "random_state": self.random_state,
            "warm_start": self.warm_start,
            "average": self.average,
        }
