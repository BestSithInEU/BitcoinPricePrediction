from .base import BaseRegressor
from sklearn.ensemble import AdaBoostRegressor


class AdaBoostRegressorModel(BaseRegressor, AdaBoostRegressor):
    """
    AdaBoostRegressorModel is a regression model that combines multiple weak regressors into a strong
    ensemble model using the AdaBoost algorithm.

    Parameters
    ----------
        n_estimators (int): The maximum number of estimators at which boosting is terminated. Default is 50.
        learning_rate (float): The learning rate shrinks the contribution of each regressor by the learning_rate.
                              Default is 1.0.
        loss (str): The loss function to use for the individual regressors. Options are 'linear', 'square',
                    and 'exponential'. Default is 'linear'.

    Methods
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the AdaBoostRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the AdaBoostRegressorModel.
    """

    def __init__(
        self,
        n_estimators=50,
        learning_rate=1.0,
        loss="linear",
    ):
        AdaBoostRegressor.__init__(
            self,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
        )
        self.name = "AdaBoostRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the AdaBoostRegressorModel using grid search and cross-validation.

        Parameters
        ----------
            X_train (pd.DataFrame): The training features.
            X_val (pd.DataFrame): The validation features.
            y_train (pd.Series): The training target.
            y_val (pd.Series): The validation target.

        Returns
        -------
            dict: The best hyperparameters found during tuning.
        """

        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1],
            "loss": ["linear", "square", "exponential"],
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the AdaBoostRegressorModel.

        Parameters
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict: The current hyperparameters of the AdaBoostRegressorModel.
        """

        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
        }
