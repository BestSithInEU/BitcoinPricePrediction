from keras.models import (
    load_model,
    Sequential,
)
from sklearn.base import BaseEstimator
import os
import joblib


def save_trained_models(trained_models, root_dir):
    for model_dict in trained_models:
        i = model_dict["step"]
        model = model_dict["model"]

        # Check model type and save accordingly
        if isinstance(model, Sequential):
            # Keras model
            model_dir = os.path.join(root_dir, model.model_name, str(i))
        elif isinstance(model, BaseEstimator):
            # Scikit-learn model
            model_dir = os.path.join(root_dir, model.model_name, str(i))
        else:
            print(f"Unsupported model type: {type(model)}")
            continue

        os.makedirs(model_dir, exist_ok=True)
        filename = os.path.join(
            model_dir, f"{i}.pkl" if isinstance(model, BaseEstimator) else f"{i}.h5"
        )

        if isinstance(model, Sequential):
            model.save(filename)
        elif isinstance(model, BaseEstimator):
            joblib.dump(model, filename)


def load_trained_models(steps, root_dir):
    loaded_models = []
    for i in steps:
        for model_type in ["Sequential", "BaseEstimator"]:
            model_dir = os.path.join(root_dir, model_type, str(i))
            filename = os.path.join(
                model_dir, f"{i}.pkl" if model_type == "BaseEstimator" else f"{i}.h5"
            )
            if os.path.isfile(filename):
                # Load model
                model = (
                    load_model(filename)
                    if model_type == "Sequential"
                    else joblib.load(filename)
                )
                loaded_models.append({"step": i, "model": model})
                break
            elif model_type == "BaseEstimator":
                print(f"No model found for step {i}")
    return loaded_models
