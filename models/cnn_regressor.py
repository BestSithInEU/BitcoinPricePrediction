from .base import BaseModelNN
from datetime import datetime
from keras import regularizers
from keras_tuner import (
    Hyperband,
    HyperModel,
)
from keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping
from keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    MaxPooling1D,
)
from keras.models import Sequential
from keras.optimizers import Adam


class CNNRegressor(HyperModel, BaseModelNN):
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
            f"CNN_Regressor_Model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        self.model = self.build_model()

    def build_model(self):
        model_name = f"cnn_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model = Sequential(name=model_name)
        model.add(
            Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                input_shape=(self.n_features, 1),
                kernel_regularizer=regularizers.l2(0.01),
                name="conv1d",
            )
        )
        model.add(MaxPooling1D(pool_size=2, name="maxpooling1d"))
        model.add(Flatten(name="flatten"))
        model.add(
            Dense(
                50,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.01),
                name="dense_1",
            )
        )
        model.add(Dense(1, name="dense_2"))
        model.compile(optimizer="adam", loss="mse")
        return model

    def tune_model_with_window_cnn(
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
                Conv1D(
                    filters=hp.Int("filters", min_value=32, max_value=256, step=32),
                    kernel_size=hp.Int("kernel_size", min_value=2, max_value=5, step=1),
                    activation="relu",
                    input_shape=(self.n_features, 1),
                )
            )
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(
                Dense(
                    units=hp.Int("dense_units", min_value=32, max_value=512, step=32),
                    activation="relu",
                )
            )
            model.add(Dense(1, activation="linear"))
            model.compile(
                optimizer=Adam(
                    learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
                ),
                loss="mean_squared_error",
            )
            model.model_name = "CNN Regressor Model"
            return model

        tuner = Hyperband(
            build_model,
            objective="val_loss",
            max_epochs=self.max_epochs,
            hyperband_iterations=self.hyperband_iterations,
            directory="models/keras_tuning/cnn",
            project_name="cnn_tuning_" + datetime.now().strftime("%Y%m%d"),
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
        history = best_model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.max_batch_size,
            epochs=self.max_epochs,
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
