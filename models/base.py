from joblib import (
    dump,
    load,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging


class BaseModelNN:
    def print_metrics(self, y_true, y_pred, n_features):
        print(f"MSE: {mean_squared_error(y_true, y_pred)}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred))}")
        print(f"MAE: {mean_absolute_error(y_true, y_pred)}")
        r2 = r2_score(y_true, y_pred)
        print(f"R^2: {r2}")
        n = len(y_true)
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
        print(f"Adjusted R^2: {adjusted_r2}")
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print(f"MAPE: {mape}%")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # plotting residual plots
        axes[0].scatter(y_pred, y_pred - y_true)
        axes[0].set_title("Residual plot")
        axes[0].set_xlabel("Predicted values")
        axes[0].set_ylabel("Residuals")

        # Actual vs Predicted values plot
        axes[1].scatter(y_true, y_pred)
        axes[1].set_title("Actual vs Predicted values")
        axes[1].set_xlabel("Actual values")
        axes[1].set_ylabel("Predicted values")

        plt.tight_layout()
        plt.show()

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(filename):
        return tf.keras.models.load_model(filename)

    def predict(self, X):
        return self.model.predict(X)

    def fit_model(
        self, X, y, epochs=10, batch_size=32, verbose=1, validation_split=0.2
    ):
        self.history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
        )


class BaseRegressor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_model(self, file_path):
        dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = load(file_path)

    def tune_model(self, X_train, X_val, y_train, y_val, param_grid, model):
        self.logger.info(f"Tuning {model.name}...")
        self.grid_search = self.grid_search_cv(model, param_grid, X_train, y_train)
        best_params = self.grid_search.best_params_
        self.logger.info(f"Best parameters: {best_params}")
        self.log_cv_score()
        self.log_validation_score(X_val, y_val)
        return self.grid_search.best_estimator_, best_params

    def grid_search_cv(self, model, param_grid, X_train, y_train):
        return GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
            n_jobs=-1,
            verbose=1,
        ).fit(X_train, y_train)

    def log_cv_score(self):
        cv_score = np.sqrt(-self.grid_search.best_score_)
        self.logger.info(f"Best cross-validation score: {cv_score}")

    def log_validation_score(self, X_val, y_val):
        val_score = np.sqrt(-self.grid_search.score(X_val, y_val))
        self.logger.info(f"Validation score: {val_score}")
