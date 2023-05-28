from keras.models import (
    load_model,
    Sequential,
)
from sklearn.base import BaseEstimator
import os
import joblib
import re


def get_val_metrics_from_log(log_file):
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
    val_metrics = get_val_metrics_from_log(log_file)
    trained_models = []
    for model_name in os.listdir(root_dir):
        model_dir = os.path.join(root_dir, model_name)
        for step_dir in os.listdir(model_dir):
            step_dir_path = os.path.join(model_dir, step_dir)
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

                # Parse step and validation metric from the filename
                base_name = os.path.basename(filename)
                step = int(base_name.split("_")[0])

                val_metric = val_metrics.get((model_name, step), None)
                if val_metric is not None:
                    trained_models.append(
                        {"step": step, "model": model, "val_metric": val_metric}
                    )

    return trained_models
