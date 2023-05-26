from joblib import (
    dump,
    load,
)
from keras.models import (
    load_model,
)
from models import BaseModelNN, BaseRegressor
from sklearn.base import BaseEstimator
import os
import pickle


def save_trained_models(trained_models, root_dir):
    for i, model in enumerate(trained_models):
        model_dir = os.path.join(root_dir, type(model).__name__)
        os.makedirs(model_dir, exist_ok=True)

        # Check model type and save accordingly
        if isinstance(model, BaseEstimator) or isinstance(model, BaseRegressor):
            # Scikit-learn model
            filename = os.path.join(model_dir, f"{i}.pkl")
            dump(model, filename)
        if isinstance(model, BaseModelNN):
            # Keras model
            sub_dir = os.path.join(model_dir, model.model_name)
            os.makedirs(sub_dir, exist_ok=True)
            filename = os.path.join(sub_dir, f"{i}.h5")
            model.save_model(filename)
        else:
            print(f"Unsupported model type: {type(model)}")


def load_sklearn_model(filename):
    model = load(filename)
    return model


def load_keras_model(filename):
    model = load_model(filename, compile=False)
    return model


def load_trained_models(root_dir):
    trained_models = []

    model_dirs = [
        dir
        for dir in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, dir))
    ]

    for model_dir in model_dirs:
        full_model_dir = os.path.join(root_dir, model_dir)
        model_files = os.listdir(full_model_dir)

        for model_file in model_files:
            model_filepath = os.path.join(full_model_dir, model_file)

            if model_filepath.endswith(".pkl"):
                # Load scikit-learn model
                model = load_sklearn_model(model_filepath)
                trained_models.append(model)
            elif model_filepath.endswith(".h5"):
                # Load Keras model
                model = load_keras_model(model_filepath)
                trained_models.append(model)

    return trained_models


def load_worker_model(worker_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_filename = f"model_{worker_id}.pkl"
    model_filepath = os.path.join(save_dir, model_filename)
    with open(model_filepath, "rb") as file:
        model = pickle.load(file)
    return model


def save_worker_model(worker_id, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_filename = f"model_{worker_id}.pkl"
    model_filepath = os.path.join(save_dir, model_filename)
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)
