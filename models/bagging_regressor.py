from .base import BaseRegressor
from sklearn.ensemble import BaggingRegressor


class BaggingRegressorModel(BaseRegressor, BaggingRegressor):
    """
    BaggingRegressorModel is a regression model that fits multiple base regressors on different subsets of the
    training data and aggregates their predictions to make the final prediction.

    Parameters
    ----------
        estimator (object): The base estimator to use for fitting on the subsets of the data. If None, the base
                            estimator is a decision tree. Default is None.
        n_estimators (int): The number of base estimators to use. Default is 10.
        random_state (int): The seed used by the random number generator. Default is 95.

    Methods
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the BaggingRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the BaggingRegressorModel.
    """

    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        random_state=95,
    ):
        BaggingRegressor.__init__(
            self,
            base_estimator=estimator,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.name = "BaggingRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the BaggingRegressorModel using grid search and cross-validation.

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
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the BaggingRegressorModel.

        Parameters
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict: The current hyperparameters of the BaggingRegressorModel.
        """
        return {
            "estimator": self.base_estimator,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
