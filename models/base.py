from joblib import (
    dump,
    load,
)
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf
import logging


class BaseModelNN:
    """
    BaseModelNN is a base class for neural network models.

    Methods:
    ----------
        save_model(filename):
            Saves the model to a file.

        load_model(filename):
            Loads the model from a file.

        predict(X):
            Generates predictions for the input data.

        fit_model(X, y, epochs=10, batch_size=32, verbose=1, validation_split=0.2):
            Fits the model to the training data.

    """

    def save_model(self, filename):
        """
        Saves the model to a file.

        Parameters:
        ----------
            filename (str): The name of the file to save the model.
        """

        self.model.save(filename)

    def load_model(filename):
        """
        Loads the model from a file.

        Parameters:
        ----------
            filename (str): The name of the file to load the model from.

        Returns:
        ----------
            tf.keras.Model: The loaded model.
        """

        return tf.keras.models.load_model(filename)

    def predict(self, X):
        """
        Generates predictions for the input data.

        Parameters:
        ----------
            X (numpy.ndarray): The input data.

        Returns:
        ----------
            numpy.ndarray: The predicted values.
        """

        return self.model.predict(X)

    def fit_model(
        self, X, y, epochs=10, batch_size=32, verbose=1, validation_split=0.2
    ):
        """
        Fits the model to the training data.

        Parameters:
        ----------
            X (numpy.ndarray): The training features.
            y (numpy.ndarray): The training target.
            epochs (int): The number of epochs to train the model. Default is 10.
            batch_size (int): The batch size for training. Default is 32.
            verbose (int): Verbosity mode. 0 - silent, 1 - progress bar, 2 - one line per epoch. Default is 1.
            validation_split (float): The fraction of the training data to use for validation. Default is 0.2.
        """

        self.history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
        )


class BaseRegressor:
    """
    BaseRegressor is a base class for regression models.

    Attributes:
    ----------
        logger (logging.Logger): The logger object for logging messages.

    Methods:
    -------
        save_model(file_path):
            Saves the model to a file using joblib.

        load_model(file_path):
            Loads the model from a file using joblib.

        tune_model(X_train, X_val, y_train, y_val, param_grid, model):
            Tunes the hyperparameters of the model using grid search and cross-validation.

        grid_search_cv(model, param_grid, X_train, y_train):
            Performs grid search cross-validation.

        log_cv_score():
            Logs the best cross-validation score.

        log_validation_score(X_val, y_val):
            Logs the validation score.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_model(self, file_path):
        """
        Saves the model to a file using joblib.

        Parameters:
        ----------
            file_path (str): The path to save the model.
        """

        dump(self.model, file_path)

    def load_model(self, file_path):
        """
        Loads the model from a file using joblib.

        Parameters:
        ----------
            file_path (str): The path to load the model from.
        """

        self.model = load(file_path)

    def tune_model(self, X_train, X_val, y_train, y_val, param_grid, model):
        """
        Tunes the hyperparameters of the model using grid search and cross-validation.

        Parameters:
        ----------
            X_train (numpy.ndarray): The training features.
            X_val (numpy.ndarray): The validation features.
            y_train (numpy.ndarray): The training target.
            y_val (numpy.ndarray): The validation target.
            param_grid (dict): The dictionary of hyperparameter values to search.
            model (object): The model object to tune.

        Returns:
        -------
            tuple: A tuple containing the best estimator and the best parameters found during tuning.
        """

        self.logger.info(f"Tuning {model.name}...")
        self.grid_search = self.grid_search_cv(model, param_grid, X_train, y_train)
        best_params = self.grid_search.best_params_
        self.logger.info(f"Best parameters: {best_params}")
        self.log_cv_score()
        self.log_validation_score(X_val, y_val)
        return self.grid_search.best_estimator_, best_params

    def grid_search_cv(self, model, param_grid, X_train, y_train):
        """
        Performs grid search cross-validation.

        Parameters:
        ----------
            model (object): The model object to tune.
            param_grid (dict): The dictionary of hyperparameter values to search.
            X_train (numpy.ndarray): The training features.
            y_train (numpy.ndarray): The training target.

        Returns:
        -------
            sklearn.model_selection.GridSearchCV: The grid search cross-validation object.
        """

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
        """
        Logs the best cross-validation score.
        """

        cv_score = -self.grid_search.best_score_
        self.logger.info(f"Best cross-validation score (MSE): {cv_score}")

    def log_validation_score(self, X_val, y_val):
        """
        Logs the validation score.

        Parameters:
        ----------
            X_val (numpy.ndarray): The validation features.
            y_val (numpy.ndarray): The validation target.
        """

        val_score = -self.grid_search.score(X_val, y_val)
        self.logger.info(f"Validation score (MSE): {val_score}")
