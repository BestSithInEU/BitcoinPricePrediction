from .base import BaseRegressor
from sklearn.neighbors import KNeighborsRegressor


class KNNRegressorModel(BaseRegressor, KNeighborsRegressor):
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
        param_grid = {
            "n_neighbors": [3, 5, 10, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],  # 1 for manhattan_distance, 2 for euclidean_distance
        }
        return super().tune_model(X_train, X_val, y_train, y_val, param_grid, self)

    def get_params(self, deep=True):
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "p": self.p,
        }
