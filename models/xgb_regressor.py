from .base import BaseRegressor
from xgboost import XGBRegressor


class XGBRegressorModel(BaseRegressor, XGBRegressor):
    """
    XGBRegressorModel is a regression model based on the eXtreme Gradient Boosting (XGBoost) algorithm.

    Parameters
    ----------
        n_estimators (int): The number of boosting iterations. Default is 100.
        learning_rate (float): The learning rate of the boosting process. Default is 0.1.
        max_depth (int): The maximum depth of each tree. Default is 3.
        min_child_weight (int): The minimum sum of instance weight needed in a child. Default is 1.
        gamma (float): The minimum loss reduction required to make a further partition on a leaf node. Default is 0.
        subsample (float): The subsample ratio of the training instances. Default is 1.
        colsample_bytree (float): The subsample ratio of columns when constructing each tree. Default is 1.

    Methods
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the XGBRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the XGBRegressorModel.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=1,
        colsample_bytree=1,
    ):
        XGBRegressor.__init__(
            self,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )
        self.name = "eXtremeGradientBoostingRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the XGBRegressorModel using grid search and cross-validation.

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
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10, 15],
            "min_child_weight": [1, 2, 5],
            "gamma": [0, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the XGBRegressorModel.

        Parameters
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict: The current hyperparameters of the XGBRegressorModel.
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }
