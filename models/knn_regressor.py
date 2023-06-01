from .base import BaseRegressor
from sklearn.neighbors import KNeighborsRegressor


class KNNRegressorModel(BaseRegressor, KNeighborsRegressor):
    """
    KNNRegressorModel is a regression model based on the K-Nearest Neighbors algorithm.

    Parameters
    ----------
        n_neighbors (int): The number of neighbors to use. Default is 2.
        weights (str or callable): The weight function used in prediction. Default is "uniform".
        p (int): The power parameter for the Minkowski metric. Default is 1.

    Methods
    -------
        tune_model(X_train, X_val, y_train, y_val):
            Tunes the hyperparameters of the KNNRegressorModel using grid search and cross-validation.

        get_params(deep=True):
            Returns the current hyperparameters of the KNNRegressorModel.
    """

    def __init__(
        self,
        n_neighbors=2,
        weights="uniform",
        p=1,
    ):
        KNeighborsRegressor.__init__(
            self,
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
        )
        self.name = "KNNRegressorModel"
        BaseRegressor.__init__(self)

    def tune_model(self, X_train, X_val, y_train, y_val):
        """
        Tunes the hyperparameters of the KNNRegressorModel using grid search and cross-validation.

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
            "n_neighbors": [3, 5, 10, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],  # 1 for manhattan_distance, 2 for euclidean_distance
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the KNNRegressorModel.

        Parameters
        ----------
            deep (bool): If True, return the parameters of all sub-objects that are estimators.
                         If False, return only the top-level parameters. Default is True.

        Returns
        -------
            dict: The current hyperparameters of the KNNRegressorModel.
        """
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "p": self.p,
        }
