import os
import pickle
from keras.models import load_model


class ModelManager:
    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_model(self, model, project_name):
        model_path = os.path.join(self.directory, project_name)
        if hasattr(model, "save"):  # Keras Model
            model_path += ".h5"
            model.save(model_path)
        else:  # Scikit-Learn Model
            model_path += ".pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        print(f"Model saved to: {model_path}")

    def load_model(self, project_name):
        model_path_keras = os.path.join(self.directory, project_name + ".h5")
        model_path_sklearn = os.path.join(self.directory, project_name + ".pkl")
        if os.path.exists(model_path_keras):  # Keras Model
            model = load_model(model_path_keras)
            print(f"Model loaded from: {model_path_keras}")
        elif os.path.exists(model_path_sklearn):  # Scikit-Learn Model
            with open(model_path_sklearn, "rb") as f:
                model = pickle.load(f)
            print(f"Model loaded from: {model_path_sklearn}")
        else:
            print("Model not found.")
            return None
        return model

    def save_tuner(self, tuner, project_name):
        tuner_path = os.path.join(self.directory, project_name + ".pkl")
        with open(tuner_path, "wb") as f:
            pickle.dump(tuner, f)
        print(f"Tuner saved to: {tuner_path}")

    def load_tuner(self, project_name):
        tuner_path = os.path.join(self.directory, project_name + ".pkl")
        if os.path.exists(tuner_path):
            with open(tuner_path, "rb") as f:
                tuner = pickle.load(f)
            print(f"Tuner loaded from: {tuner_path}")
            return tuner
        else:
            print("Tuner not found.")
            return None
