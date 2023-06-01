from .base import BaseRegressor
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressorModel(BaseRegressor, RandomForestRegressor):
    """
    RandomForestRegressorModel is a regression model based on the Random Forest algorithm.

    Parameters
    ----------
        n_estimators (int): The number of trees in the forest. Default is 100.
        max_depth (int or None): The maximum depth of the tree. None indicates unlimited depth. Default is None.
        min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node. Default is 1.
        max_features (str or int): The number of features to consider when looking for the best split.
                                   'auto' uses sqrt(n_features), 'sqrt' uses sqrt(n_features), 'log2' uses log2(n_features),
                                   None uses n_features, and int specifies the number of features.
                                   Default is 'auto'.

    Methods
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the RandomForestRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the RandomForestRegressorModel.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="auto",
    ):
        RandomForestRegressor.__init__(
            self,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        self.name = "RandomForestRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the RandomForestRegressorModel using grid search and cross-validation.

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
            "max_depth": [5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", None],
        }

        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the RandomForestRegressorModel.

        Parameters
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict: The current hyperparameters of the RandomForestRegressorModel.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
        }
