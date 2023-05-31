from keras.models import (
    load_model,
    Sequential,
)
from sklearn.base import BaseEstimator
import os
import joblib
import re


def get_val_metrics_from_log(log_file):
    """
    Extract validation metrics from a given log file.

    Parameters:
    -----------
    log_file : str
        Path to the log file.

    Returns:
    --------
    dict
        A dictionary where the keys are tuples of (model_name, step) and the values are validation metrics.
    """

    val_metrics = {}
    with open(log_file, "r") as file:
        for line in file.readlines():
            search_result = re.search(
                r"Model: (.*), Step: (.*), Val Metric: (.*),", line
            )
            if search_result:
                model_name = search_result.group(1)
                step = int(search_result.group(2))
                val_metric = float(search_result.group(3))
                val_metrics[(model_name, step)] = val_metric
    return val_metrics


def save_trained_models(trained_models, root_dir):
    """
    Saves trained models to a specified directory.

    Parameters:
    -----------
    trained_models : list of dicts
        A list of dictionaries, where each dictionary contains details about a trained model.
    root_dir : str
        The root directory where the models will be saved.
    """

    for model_dict in trained_models:
        i = model_dict["step"]
        model = model_dict["model"]

        # Check model type and save accordingly
        if isinstance(model, Sequential):
            # Keras model
            model_dir = os.path.join(root_dir, model.name, str(i))
        elif isinstance(model, BaseEstimator):
            # Scikit-learn model
            model_dir = os.path.join(root_dir, model.name, str(i))
        else:
            print(f"Unsupported model type: {type(model)}")
            continue

        os.makedirs(model_dir, exist_ok=True)
        filename = os.path.join(
            model_dir,
            f"{i}.pkl" if isinstance(model, BaseEstimator) else f"{i}.h5",
        )

        if isinstance(model, Sequential):
            model.save(filename)
        elif isinstance(model, BaseEstimator):
            joblib.dump(model, filename)


def load_trained_models(root_dir, log_file):
    """
    Loads trained models from a specified directory. The models are filtered based on the
    validation metrics provided in a log file.

    Parameters:
    -----------
    root_dir : str
        The root directory from where the models will be loaded.
    log_file : str
        Path to the log file containing validation metrics.

    Returns:
    --------
    list
        A list of dictionaries, where each dictionary contains details about a trained model
        and its validation metric.
    """

    val_metrics = get_val_metrics_from_log(log_file)
    trained_models = []
    for model_name in os.listdir(root_dir):
        model_dir = os.path.join(root_dir, model_name)
        for step_dir in os.listdir(model_dir):
            step_dir_path = os.path.join(model_dir, step_dir)
            step = int(step_dir)
            for file in os.listdir(step_dir_path):
                filename = os.path.join(step_dir_path, file)
                if file.endswith(".h5"):
                    # Keras model
                    model = load_model(filename)
                    model_type = "keras"
                elif file.endswith(".pkl"):
                    # Scikit-learn model
                    model = joblib.load(filename)
                    model_type = "sklearn"
                else:
                    print(f"Unsupported file type: {file}")
                    continue

                val_metric = val_metrics.get((model_name, step), None)
                if val_metric is not None:
                    trained_models.append(
                        {
                            "step": step,
                            "model": model,
                            "val_metric": val_metric,
                            "model_type": model_type,
                        }
                    )

    return trained_models
