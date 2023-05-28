import os
import logging
import time
import warnings
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error
from models import BaseModelNN
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from .utils import mean_absolute_percentage_error, root_mean_squared_log_error
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from math import sqrt
from collections import Counter
import pickle


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a console handler with INFO level
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

# Create a file handler with ERROR level
log_file = "rolling_window_training.log"
f_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=1024 * 1024 * 5, backupCount=5
)  # 5 MB per file, 5 backup files
f_handler.setLevel(logging.DEBUG)
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)


# Define a new logger for model metrics
metrics_logger = logging.getLogger("metrics_logger")
metrics_logger.setLevel(logging.INFO)

# Create a file handler for it
metrics_log_file = "model_metrics.log"
metrics_f_handler = logging.FileHandler(metrics_log_file)
metrics_f_handler.setLevel(logging.INFO)

# You can use a simple formatter since this is mostly structured data
metrics_f_format = logging.Formatter("%(asctime)s - %(message)s")
metrics_f_handler.setFormatter(metrics_f_format)

# Add the handler to the logger
metrics_logger.addHandler(metrics_f_handler)


class RollingWindowTrainer:
    def __init__(
        self,
        scaler,
        lstm_stop_early,
        nn_stop_early,
        cnn_stop_early,
        X_train_val,
        y_train_val,
        train_window=100,
        val_window=20,
        step_size=5,
        checkpoint_path="models/save/checkpoints/",
        model_list=None,
    ):
        self.model_list = model_list if model_list is not None else []
        self.lstm_stop_early = lstm_stop_early
        self.nn_stop_early = nn_stop_early
        self.cnn_stop_early = cnn_stop_early
        self.X_train_val = X_train_val
        self.y_train_val = y_train_val
        self.train_window = train_window
        self.val_window = val_window
        self.step_size = step_size
        self.stop_training = False
        self.pause_training = False
        self.current_step = 0
        self.total_windows = int(
            (len(self.X_train_val) - self.train_window - self.val_window)
            / self.step_size
        )
        self.scaler = scaler
        self.all_models = []
        self.all_val_metrics = {}
        self.best_models = []
        self.best_model_info = None
        self.val_metric = float("inf")
        self.time_consumption = {}
        self.histories_nn = []
        self.histories_cnn = []
        self.histories_lstm = []
        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def start_training(self):
        if self.stop_training:
            logger.warning("Training is already stopped.")
            return

        logger.info("Training started.")

        for i in range(self.current_step, self.total_windows):
            if self.stop_training:
                logger.info(f"Training stopped at step {i}")
                break

            best_model_info_in_window = None
            best_val_metric_in_window = float("inf")

            for (
                model_class,
                model_params,
            ) in self.model_list:
                self.model = model_class(**model_params)
                model_class_name = model_class.__name__

                if model_class_name not in self.all_val_metrics:
                    self.all_val_metrics[model_class_name] = []

                while self.pause_training:
                    time.sleep(1)

                train_start, train_end, val_start, val_end = self.get_window_indices(i)

                X_train_window, y_train_window = self.get_train_window(
                    train_start, train_end
                )
                X_val_window, y_val_window = self.get_val_window(val_start, val_end)

                try:
                    start_time = time.time()
                    logger.info(f"Start training {self.model.name}")

                    if isinstance(self.model, BaseModelNN):
                        checkpoint_dir = self.get_checkpoint_dir()
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_callback = self.get_checkpoint_callback(
                            checkpoint_dir
                        )

                        (
                            tuned_model,
                            history,
                            flag,
                        ) = self.train_nn_model_with_window(
                            X_train_window,
                            y_train_window,
                            X_val_window,
                            y_val_window,
                            checkpoint_callback,
                        )
                        if flag == 1:
                            self.histories_lstm.append(history)
                        elif flag == 2:
                            self.histories_cnn.append(history)
                        elif flag == 3:
                            self.histories_nn.append(history)
                        self.check_overfitting(i, history)

                        if hasattr(self.model, "tune_model_with_window_lstm"):
                            X_val_window = np.reshape(
                                X_val_window.to_numpy(),
                                (X_val_window.shape[0], 1, X_val_window.shape[1]),
                            )
                            y_val_window = np.reshape(
                                y_val_window.to_numpy(),
                                (y_val_window.shape[0], 1),
                            )

                            val_predictions = tuned_model.predict(X_val_window)

                            if val_predictions.ndim == 3:
                                val_predictions = val_predictions.reshape(
                                    val_predictions.shape[0], val_predictions.shape[2]
                                )
                            if y_val_window.ndim == 3:
                                y_val_window = y_val_window.to_numpy().reshape(
                                    y_val_window.shape[0], y_val_window.shape[2]
                                )

                            val_predictions = val_predictions.reshape(-1, 1)
                            val_predictions = self.scaler.inverse_transform(
                                val_predictions
                            )

                            y_val_window = y_val_window.reshape(-1, 1)
                            y_val_window_inverse = self.scaler.inverse_transform(
                                y_val_window
                            )

                            val_metric = mean_squared_error(
                                y_val_window_inverse, val_predictions
                            )
                            self.all_val_metrics[model_class_name].append(val_metric)
                        else:
                            val_predictions = tuned_model.predict(X_val_window)
                            val_predictions = val_predictions.reshape(-1, 1)
                            val_predictions = self.scaler.inverse_transform(
                                val_predictions
                            )

                            y_val_window = y_val_window.to_numpy().reshape(-1, 1)
                            y_val_window_inverse = self.scaler.inverse_transform(
                                y_val_window
                            )

                            val_metric = mean_squared_error(
                                y_val_window_inverse, val_predictions
                            )
                            self.all_val_metrics[model_class_name].append(val_metric)

                    else:
                        model_requires_squeeze = [
                            "BaggingRegressorModel",
                            "GradientBoostingRegressorModel",
                            "RandomForestRegressorModel",
                            "SupportVectorRegressorModel",
                        ]

                        logger.info(f"Shape of y_train: {y_train_window.shape}")
                        logger.info(f"Shape of y_val: {y_val_window.shape}")

                        if self.model.name in model_requires_squeeze:
                            y_train_window, y_val_window = (
                                y_train_window.squeeze(),
                                y_val_window.squeeze(),
                            )
                            logger.info("Applied squeeze() on the target variable.")
                        else:
                            logger.info(
                                "Not applying squeeze() on the target variable."
                            )

                        best_model = self.tune_non_nn_model(
                            X_train_window,
                            X_val_window,
                            y_train_window,
                            y_val_window,
                        )
                        best_model.fit(X_train_window, y_train_window)

                        val_predictions = best_model.predict(X_val_window)
                        val_predictions = val_predictions.reshape(-1, 1)
                        val_predictions = self.scaler.inverse_transform(val_predictions)

                        y_val_window = y_val_window.to_numpy().reshape(-1, 1)
                        y_val_window_inverse = self.scaler.inverse_transform(
                            y_val_window
                        )

                        val_metric = mean_squared_error(
                            y_val_window_inverse, val_predictions
                        )
                        val_metric = mean_squared_error(y_val_window, val_predictions)
                        self.all_val_metrics[model_class_name].append(val_metric)

                    model_info = {
                        "step": i,
                        "model": tuned_model
                        if isinstance(self.model, BaseModelNN)
                        else best_model,
                        "val_metric": val_metric,
                    }
                    self.all_models.append(model_info)

                    if val_metric < best_val_metric_in_window:
                        best_val_metric_in_window = val_metric
                        best_model_info_in_window = model_info

                    self.update_time_consumption(start_time)

                    # Log model metrics
                    idx = self.model.name.split("_")[0]
                    metrics_logger.info(
                        f"Model: {self.model.name.split('_')[0]}, Step: {i}, Val Metric: {val_metric}, Time Consumption: {self.time_consumption[idx][i]}"
                    )
                except Exception as e:
                    logger.exception(f"Error occurred during training: {str(e)}")

                self.current_step = i
                logger.info(f"Completed training window: {i}")

            # After each window, add the best model to the list
            if best_model_info_in_window is not None:
                self.best_models.append(best_model_info_in_window)

        self.stop_training = False
        logger.info("Training complete.")

    def get_all_val_metrics(self):
        return self.all_val_metrics

    def get_window_indices(self, step):
        train_start = step * self.step_size
        train_end = train_start + self.train_window
        val_start = train_end
        val_end = val_start + self.val_window
        return train_start, train_end, val_start, val_end

    def get_train_window(self, start, end):
        X_train_window = self.X_train_val.iloc[start:end]
        y_train_window = self.y_train_val.iloc[start:end]
        return X_train_window, y_train_window

    def get_val_window(self, start, end):
        X_val_window = self.X_train_val.iloc[start:end]
        y_val_window = self.y_train_val.iloc[start:end]
        return X_val_window, y_val_window

    def get_checkpoint_dir(self):
        return os.path.join(self.checkpoint_path, self.model.name)

    def get_checkpoint_callback(self, checkpoint_dir):
        return ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.h5"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )

    def get_best_model_at_window(self, window):
        for model_info in self.best_models:
            if model_info["step"] == window:
                return model_info
        return None

    def get_best_models(self):
        return self.best_models

    def set_best_models(self, loaded_models):
        self.best_models = loaded_models

    def get_all_models(self):
        return self.all_models

    def set_all_models(self, loaded_models):
        self.all_models = loaded_models

    def print_model_info(self):
        for model_idx, model_info in enumerate(self.best_models):
            model = model_info["model"]
            step = model_info["step"]
            print(model_info)

    def get_histories(self):
        return self.histories_nn, self.histories_cnn, self.histories_lstm

    def save_histories(
        self,
        file_name_nn="nn_history",
        file_name_cnn="cnn_history",
        file_name_lstm="lstm_history",
    ):
        base_dir = "models/save/histories"
        os.makedirs(base_dir, exist_ok=True)

        if self.histories_nn is not None and file_name_nn is not None:
            with open(os.path.join(base_dir, f"{file_name_nn}.pkl"), "wb") as file:
                pickle.dump(self.histories_nn, file)
        if self.histories_cnn is not None and file_name_cnn is not None:
            with open(os.path.join(base_dir, f"{file_name_cnn}.pkl"), "wb") as file:
                pickle.dump(self.histories_cnn, file)
        if self.histories_lstm is not None and file_name_lstm is not None:
            with open(os.path.join(base_dir, f"{file_name_lstm}.pkl"), "wb") as file:
                pickle.dump(self.histories_lstm, file)

    def load_and_set_histories(
        self,
        file_name_nn="nn_history",
        file_name_cnn="cnn_history",
        file_name_lstm="lstm_history",
    ):
        base_dir = "models/save/histories"

        if file_name_nn is not None:
            file_path_nn = os.path.join(base_dir, f"{file_name_nn}.pkl")
            if os.path.exists(file_path_nn):
                with open(file_path_nn, "rb") as file:
                    self.histories_nn = pickle.load(file)
            else:
                print(f"No file found at {file_path_nn}")
        if file_name_cnn is not None:
            file_path_cnn = os.path.join(base_dir, f"{file_name_cnn}.pkl")
            if os.path.exists(file_path_cnn):
                with open(file_path_cnn, "rb") as file:
                    self.histories_cnn = pickle.load(file)
            else:
                print(f"No file found at {file_path_cnn}")
        if file_name_lstm is not None:
            file_path_lstm = os.path.join(base_dir, f"{file_name_lstm}.pkl")
            if os.path.exists(file_path_lstm):
                with open(file_path_lstm, "rb") as file:
                    self.histories_lstm = pickle.load(file)
            else:
                print(f"No file found at {file_path_lstm}")

    def train_nn_model_with_window(
        self, X_train, y_train, X_val, y_val, checkpoint_callback
    ):
        model_flag = 0
        best_hps = None
        if hasattr(self.model, "tune_model_with_window_lstm"):
            X_train_lstm = np.reshape(
                X_train.to_numpy(), (X_train.shape[0], 1, X_train.shape[1])
            )
            y_train_lstm = np.reshape(y_train.to_numpy(), (y_train.shape[0], 1))
            X_val_lstm = np.reshape(
                X_val.to_numpy(), (X_val.shape[0], 1, X_val.shape[1])
            )
            y_val_lstm = np.reshape(y_val.to_numpy(), (y_val.shape[0], 1))

            tuned_model, history, best_hps = self.model.tune_model_with_window_lstm(
                X_train_lstm,
                y_train_lstm,
                X_val_lstm,
                y_val_lstm,
                callbacks=[self.lstm_stop_early, checkpoint_callback],
            )
            model_flag = 1
        elif hasattr(self.model, "tune_model_with_window_nn"):
            tuned_model, history, best_hps = self.model.tune_model_with_window_nn(
                X_train,
                y_train,
                X_val,
                y_val,
                callbacks=[self.nn_stop_early, checkpoint_callback],
            )
            model_flag = 2
        elif hasattr(self.model, "tune_model_with_window_cnn"):
            tuned_model, history, best_hps = self.model.tune_model_with_window_cnn(
                X_train,
                y_train,
                X_val,
                y_val,
                callbacks=[self.cnn_stop_early, checkpoint_callback],
            )
            model_flag = 3

        metrics_logger.info(
            f"Best hyperparameters for {self.model.name}: {best_hps.values}"
        )

        return tuned_model, history, model_flag

    def check_overfitting(self, step, history):
        patience = 5
        if len(history.history["val_loss"]) > patience:
            if all(
                i > j
                for i, j in zip(
                    history.history["val_loss"][-patience:],
                    history.history["val_loss"][-patience - 1 : -1],
                )
            ):
                logger.warning(
                    f"{self.model.name} is overfitting on the validation data at step {step}"
                )
        self.val_metric = min(self.val_metric, history.history["val_loss"][-1])

    def tune_non_nn_model(self, X_train, X_val, y_train, y_val):
        best_model = None
        best_hps = None
        try:
            best_model, best_hps = self.model.tune_model(X_train, X_val, y_train, y_val)
            metrics_logger.info(
                f"Best hyperparameters for {self.model.name}: {best_hps}"
            )
        except Exception as e:
            logger.error(f"Error occurred during model tuning: {str(e)}")
            best_model = None

        return best_model

    def update_time_consumption(self, start_time):
        end_time = time.time()
        time_taken = end_time - start_time
        name = self.model.name.split("_")[0]  # Strip the timestamp

        if name not in self.time_consumption:
            self.time_consumption[name] = [time_taken]
        else:
            self.time_consumption[name].append(time_taken)

    def predict_best_models(self, X):
        """
        Predicts the target variable given input features, using the best models.

        Args:
        X (pd.DataFrame or np.array): The input features.

        Returns:
        A dataframe of predictions from the best models.
        """

        all_predictions = {}
        for model_idx, model_info in enumerate(self.best_models):
            model = model_info["model"]
            step = model_info["step"]

            # Adjust X if necessary based on the model's training window
            window_start = max(
                0, len(X) - self.train_window * (self.total_windows - step)
            )
            X_window = X[window_start:]

            name = f"{model.name}_{step}"
            if name.lower().startswith("lstmregressormodel"):
                X_window = np.reshape(
                    X_window.to_numpy(), (X_window.shape[0], 1, X_window.shape[1])
                )
                all_predictions[name] = model.predict(X_window)
            else:
                all_predictions[name] = model.predict(X_window)

        # Convert to DataFrame if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            for name in all_predictions:
                model_predictions = np.array(all_predictions[name]).flatten()
                all_predictions[name] = pd.Series(
                    model_predictions, index=X.index[window_start:]
                )

            all_predictions = (
                pd.DataFrame(all_predictions)
                .mean(axis=1)
                .to_frame()
                .rename(columns={0: "Predictions"})
            )

        return all_predictions

    def weighted_predict_best_models(self, X):
        all_predictions = {}
        model_weights = {}
        total_weights = 0

        for model_idx, model_info in enumerate(self.best_models):
            model = model_info["model"]
            step = model_info["step"]

            # Adjust X if necessary based on the model's training window
            window_start = max(
                0, len(X) - self.train_window * (self.total_windows - step)
            )
            X_window = X[window_start:]

            name = f"{model.name}_{step}"
            if name.lower().startswith("lstmregressormodel"):
                X_window = np.reshape(
                    X_window.to_numpy(), (X_window.shape[0], 1, X_window.shape[1])
                )
                all_predictions[name] = model.predict(X_window)
            else:
                all_predictions[name] = model.predict(X_window)

            # Use inverse of validation error as weight
            model_weights[name] = 1.0 / model_info["val_metric"]
            total_weights += model_weights[name]

        # Normalize weights
        for name in model_weights:
            model_weights[name] /= total_weights

        # Convert to DataFrame if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            weighted_predictions = pd.DataFrame()
            for name in all_predictions:
                model_predictions = np.array(all_predictions[name]).flatten()
                weighted_model_prediction = pd.Series(
                    model_predictions * model_weights[name],
                    index=X.index[window_start:],
                )
                weighted_predictions = pd.concat(
                    [weighted_predictions, weighted_model_prediction], axis=1
                )

            # Sum all the weighted predictions
            weighted_predictions["Predictions"] = weighted_predictions.sum(axis=1)
            weighted_predictions = weighted_predictions[["Predictions"]]

        else:
            # Weighted averaging of predictions
            weighted_predictions = None
            for name, prediction in all_predictions.items():
                if weighted_predictions is None:
                    weighted_predictions = model_weights[name] * prediction
                else:
                    weighted_predictions += model_weights[name] * prediction

        return weighted_predictions

    def vote_predict_best_models(self, X):
        all_predictions = {}
        for model_idx, model_info in enumerate(self.best_models):
            model = model_info["model"]
            step = model_info["step"]

            # Adjust X if necessary based on the model's training window
            window_start = max(
                0, len(X) - self.train_window * (self.total_windows - step)
            )
            X_window = X[window_start:]

            name = f"{model.name}_{step}"
            if name.lower().startswith("lstmregressormodel"):
                X_window = np.reshape(
                    X_window.to_numpy(), (X_window.shape[0], 1, X_window.shape[1])
                )
                all_predictions[name] = model.predict(X_window)
            else:
                all_predictions[name] = model.predict(X_window)

        # Perform voting
        vote_predictions = []
        for i in range(X.shape[0]):
            model_votes = []
            for prediction in all_predictions.values():
                vote = prediction[i]
                if isinstance(vote, np.ndarray) and vote.size == 1:
                    vote = vote.item()  # convert single-element arrays to scalars
                model_votes.append(vote)
            vote_predictions.append(Counter(model_votes).most_common(1)[0][0])

        # Convert to DataFrame if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            vote_predictions_series = pd.Series(
                vote_predictions, index=X.index[window_start:]
            )
            all_predictions = vote_predictions_series.to_frame().rename(
                columns={0: "Predictions"}
            )

        return all_predictions

    def predict_all_models(self, X):
        """
        Predicts the target variable given input features, using all trained models.

        Args:
        X (pd.DataFrame or np.array): The input features.

        Returns:
        The mean of predictions from all models.
        """

        all_predictions = {}
        for model_idx, model_info in enumerate(self.all_models):
            model = model_info["model"]
            step = model_info["step"]

            # Adjust X if necessary based on the model's training window
            window_start = max(
                0, len(X) - self.train_window * (self.total_windows - step)
            )
            X_window = X[window_start:]

            name = f"{model.name}_{step}"
            if name.lower().startswith("lstmregressormodel"):
                X_window = np.reshape(
                    X_window.to_numpy(), (X_window.shape[0], 1, X_window.shape[1])
                )
                all_predictions[name] = model.predict(X_window)
            else:
                all_predictions[name] = model.predict(X_window)

        # Convert to DataFrame if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            for name in all_predictions:
                # Check if predictions are in a nested list form, if yes flatten them
                if isinstance(all_predictions[name][0], list) or isinstance(
                    all_predictions[name][0], np.ndarray
                ):
                    all_predictions[name] = [
                        item for sublist in all_predictions[name] for item in sublist
                    ]
                model_predictions = np.array(all_predictions[name]).flatten()
                all_predictions[name] = pd.Series(
                    model_predictions, index=X.index[window_start:]
                )

            all_predictions = pd.DataFrame(all_predictions)
            df_predictions = pd.DataFrame(all_predictions)

            # Create a mapping of model names to their columns
            model_column_groups = {}
            for col in df_predictions.columns:
                name = col.split("_")[
                    0
                ]  # Adjust this as necessary to correctly identify the model name
                if name not in model_column_groups:
                    model_column_groups[name] = []
                model_column_groups[name].append(col)

            # Calculate mean for each model
            for name, cols in model_column_groups.items():
                df_predictions[name + " mean"] = df_predictions[cols].mean(axis=1)

            # Drop the original columns
            df_predictions = df_predictions[
                [col for col in df_predictions.columns if " mean" in col]
            ]

        return df_predictions

    def generate_mean_df(self, predictions_all):
        total_columns = predictions_all.shape[1]
        columns_per_model = total_columns // self.total_windows

        mean_df = pd.DataFrame()
        for i in range(columns_per_model):
            start = i * self.total_windows
            end = (i + 1) * self.total_windows
            model_df = predictions_all.iloc[:, start:end]
            mean_df[f"model_{i+1}_mean"] = model_df.mean(axis=1)

        return mean_df

    @staticmethod
    def evaluate_with_rolling_window(X_test, y_test, step_size, best_model):
        try:
            predictions = RollingWindowTrainer.predict_with_rolling_window(
                X_test, step_size, best_model
            )
            y_test_flattened = y_test.values.flatten()

            min_length = min(len(predictions), len(y_test_flattened))
            predictions = predictions[:min_length]
            y_test_flattened = y_test_flattened[:min_length]

            test_metrics = []

            mae = mean_absolute_error(y_test_flattened, predictions)
            mse = mean_squared_error(y_test_flattened, predictions)
            r2 = r2_score(y_test_flattened, predictions)
            rmse = sqrt(mse)
            metrics = {
                "Model": best_model.name,
                "Test MAE (Mean Absolute Error)": mae,
                "Test MSE (Mean Squared Error)": mse,
                "Test R2 Score (Coefficient of Determination)": r2,
                "Test RMSE (Root Mean Square Deviation)": rmse,
            }
            test_metrics.append(metrics)

            df = pd.DataFrame(test_metrics)
            logger.info(f"Test evaluation metrics: \n{df}")
            return df

        except Exception as e:
            logger.error(f"Error occurred during evaluation: {str(e)}")
            return pd.DataFrame()
