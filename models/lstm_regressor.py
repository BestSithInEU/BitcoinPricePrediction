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
    LSTM,
)
from keras.models import Sequential
from keras.optimizers import Adam


class LSTMRegressor(HyperModel, BaseModelNN):
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
        self.name = "LstmRegressorModel"
        self.model = self.build_model()

    def build_model(self):
        model = Sequential(name="LstmRegressorModel")
        model.add(
            LSTM(
                100,
                activation="tanh",
                return_sequences=True,
                input_shape=(1, self.n_features),
                kernel_regularizer=regularizers.l2(0.01),
                name="lstm_1",
            )
        )
        model.add(
            LSTM(
                50,
                activation="tanh",
                return_sequences=True,
                kernel_regularizer=regularizers.l2(0.01),
                name="lstm_2",
            )
        )
        model.add(
            LSTM(
                50,
                activation="tanh",
                kernel_regularizer=regularizers.l2(0.01),
                name="lstm_3",
            )
        )
        model.add(Dropout(0.5, name="dropout"))
        model.add(Dense(1, name="dense"))
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(),
        )
        return model

    def tune_model_with_window_lstm(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        callbacks=None,
    ):
        def build_model(hp):
            model = Sequential(name="LstmRegressorModel")
            model.add(
                LSTM(
                    units=hp.Int("units", min_value=32, max_value=512, step=32),
                    activation="tanh",
                    return_sequences=True,
                    input_shape=(1, self.n_features),
                )
            )
            model.add(
                LSTM(
                    units=hp.Int("units", min_value=32, max_value=512, step=32),
                    activation="tanh",
                )
            )
            model.add(
                Dropout(
                    hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
                )
            )
            model.add(Dense(1, activation="linear"))
            model.compile(
                optimizer=Adam(
                    learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
                ),
                loss="mean_squared_error",
            )
            return model

        tuner = Hyperband(
            build_model,
            objective="val_loss",
            max_epochs=self.max_epochs,
            hyperband_iterations=self.hyperband_iterations,
            directory="models/keras_tuning/lstm",
            project_name="lstm_tuning_" + datetime.now().strftime("%Y%m%d"),
        )

        tuner.search_space_summary()

        if X_val is not None and y_val is not None:
            tuner.search(
                x=X_train,
                y=y_train,
                epochs=self.tuner_epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )
        else:
            tuner.search(
                x=X_train,
                y=y_train,
                epochs=self.tuner_epochs,
                callbacks=callbacks,
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
        return {
            "n_features": self.n_features,
            "max_epochs": self.max_epochs,
            "hyperband_iterations": self.hyperband_iterations,
            "patience": self.patience,
            "tuner_epochs": self.tuner_epochs,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
