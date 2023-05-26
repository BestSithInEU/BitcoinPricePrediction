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
        self.model_name = (
            f"NN_Regressor_Model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        self.model = self.build_model()

    def build_model(self):
        model_name = f"nn_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model = Sequential(name=model_name)
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
        def build_model(hp):
            model = Sequential()
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
            model.model_name = "NN Regressor Model"
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
        history = best_model.fit(
            x=X_train,
            y=y_train,
            epochs=self.max_epochs,
            batch_size=self.max_batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )

        return best_model, history

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
