from .base import BaseModelNN
from datetime import datetime
from keras import regularizers
from keras_tuner import (
    Hyperband,
    HyperModel,
)
from keras.callbacks import EarlyStopping
from keras.layers import (
    Dense,
    Dropout,
)
from keras.models import Sequential
from keras.optimizers import Adam


class NeuralNetworkRegressor(HyperModel, BaseModelNN):
    """
    NeuralNetworkRegressor is a regression model based on a neural network.

    Parameters
    ----------
        n_features (int): The number of input features.
        max_epochs (int): The maximum number of epochs to train the model. Default is 5.
        max_batch_size (int): The maximum batch size for training. Default is 5.
        hyperband_iterations (int): The number of Hyperband iterations. Default is 1.
        patience (int): The number of epochs with no improvement after which training will be stopped if early stopping is used. Default is 5.
        tuner_epochs (int): The number of epochs to train the tuner. Default is 5.

    Methods
    -------
        build_model():
            Builds and compiles the neural network model.

        tune_model_with_window_nn(X_train, y_train, X_val, y_val, callbacks=None):
            Tunes the hyperparameters of the NeuralNetworkRegressor using Hyperband tuner.

        get_params(deep=True):
            Returns the current hyperparameters of the NeuralNetworkRegressor.

        set_params(**parameters):
            Sets the value of the specified hyperparameters.

    Attributes
    ----------
        n_features (int): The number of input features.
        max_epochs (int): The maximum number of epochs to train the model.
        max_batch_size (int): The maximum batch size for training.
        hyperband_iterations (int): The number of Hyperband iterations.
        patience (int): The number of epochs with no improvement after which training will be stopped if early stopping is used.
        tuner_epochs (int): The number of epochs to train the tuner.
        name (str): The name of the NeuralNetworkRegressor model.
        model (Sequential): The compiled neural network model.
    """

    def __init__(
        self,
        n_features,
        max_epochs=5,
        max_batch_size=5,
        hyperband_iterations=1,
        patience=5,
        tuner_epochs=5,
    ):
        self.n_features = n_features
        self.max_epochs = max_epochs
        self.max_batch_size = max_batch_size
        self.hyperband_iterations = hyperband_iterations
        self.patience = patience
        self.tuner_epochs = tuner_epochs
        self.name = "NeuralNetworkRegressor"
        self.model = self.build_model()

    def build_model(self):
        """
        Builds and compiles the neural network model.

        Returns
        -------
            Sequential: The compiled neural network model.
        """
        model = Sequential(name="NeuralNetworkRegressor")
        model.add(
            Dense(
                128,
                input_dim=self.n_features,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.01),
                name="dense_1",
            )
        )
        model.add(Dropout(0.3, name="dropout1"))
        model.add(
            Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.01),
                name="dense_2",
            )
        )
        model.add(Dropout(0.3, name="dropout2"))
        model.add(
            Dense(
                32,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.01),
                name="dense_3",
            )
        )
        model.add(Dropout(0.3, name="dropout3"))
        model.add(Dense(1, name="dense4"))
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(),
        )
        return model

    def tune_model_with_window_nn(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        callbacks=None,
    ):
        """
        Tunes the hyperparameters of the NeuralNetworkRegressor using Hyperband tuner.

        Parameters
        ----------
            X_train (array-like): The training input samples.
            y_train (array-like): The target values for the training samples.
            X_val (array-like): The validation input samples.
            y_val (array-like): The target values for the validation samples.
            callbacks (list): List of Keras callbacks. Default is None.

        Returns
        -------
            tuple: A tuple containing the best model, training history, and best hyperparameters.
        """

        def build_model(hp):
            model = Sequential(name="NeuralNetworkRegressor")
            model.add(
                Dense(
                    hp.Int("units_input", 32, 256, 32),
                    input_dim=self.n_features,
                    activation="relu",
                )
            )
            model.add(
                Dropout(
                    hp.Float("dropout_rate_1", min_value=0.1, max_value=0.5, step=0.1),
                )
            )
            model.add(
                Dense(hp.Int("units_hidden", 32, 256, 32), activation="relu"),
            )
            model.add(
                Dropout(
                    hp.Float("dropout_rate_2", min_value=0.1, max_value=0.5, step=0.1),
                )
            )
            model.add(Dense(1, activation="linear"))
            model.compile(
                loss="mean_squared_error",
                optimizer=Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
            )
            return model

        tuner = Hyperband(
            build_model,
            objective="val_loss",
            max_epochs=self.max_epochs,
            hyperband_iterations=self.hyperband_iterations,
            directory="models/keras_tuning/nn",
            project_name="lstm_tuning_" + datetime.now().strftime("%Y%m%d"),
        )

        tuner.search_space_summary()

        stop_early = EarlyStopping(monitor="val_loss", patience=self.patience)

        if X_val is not None and y_val is not None:
            tuner.search(
                x=X_train,
                y=y_train,
                epochs=self.tuner_epochs,
                validation_data=(X_val, y_val),
                callbacks=[stop_early],
            )
        else:
            tuner.search(
                x=X_train,
                y=y_train,
                epochs=self.tuner_epochs,
                callbacks=[stop_early],
            )

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        history = best_model.fit(
            x=X_train,
            y=y_train,
            epochs=self.max_epochs,
            batch_size=self.max_batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )

        return best_model, history, best_hps

    def get_params(self, deep=True):
        """
        Returns the current hyperparameters of the NeuralNetworkRegressor.

        Returns
        -------
            dict: A dictionary of the current hyperparameters.
        """
        return {
            "n_features": self.n_features,
            "max_epochs": self.max_epochs,
            "hyperband_iterations": self.hyperband_iterations,
            "patience": self.patience,
            "tuner_epochs": self.tuner_epochs,
        }

    def set_params(self, **parameters):
        """
        Sets the value of the specified hyperparameters.

        Returns
        -------
            self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
