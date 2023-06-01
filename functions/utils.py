from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_squared_log_error,
    mean_squared_error,
)

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
from keras.utils import plot_model
from PIL import Image


def calculate_metrics(preds, y_test):
    """
    Calculates Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for the given predictions and actual values.
    It also calculates daily MSE and RMSE and stores them in a pandas DataFrame.

    Parameters
    ----------
    preds : np.ndarray or pd.Series
        The predicted values.
    y_test : np.ndarray or pd.Series
        The actual values.

    Returns
    -------
    mse : float
        The Mean Squared Error.
    rmse : float
        The Root Mean Squared Error.
    metrics_df : pd.DataFrame
        A dataframe containing daily MSE and RMSE.
    """

    # Check if the inputs are numpy arrays and convert them to pandas Series
    if isinstance(preds, np.ndarray):
        preds = pd.Series(preds.flatten())
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test.flatten())

    # Compute MSE and RMSE
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    # Compute daily MSE and RMSE
    daily_mse = (y_test - preds) ** 2
    daily_rmse = np.sqrt(daily_mse)

    # Create a DataFrame to store the results
    metrics_df = pd.DataFrame(
        {
            "MSE": daily_mse,
            "RMSE": daily_rmse,
        }
    )

    return mse, rmse, metrics_df


def resize_and_remove_background(image_path, output_path, size=(800, 800)):
    """
    Resizes an image to the specified size, removes its background, and saves it in the specified output path.

    Parameters
    ----------
    image_path : str
        The path to the input image.
    output_path : str
        The path to save the output image.
    size : tuple of int, optional
        The desired output size. Default is (800, 800).
    """

    # Open an image file
    with Image.open(image_path) as img:
        # Resize the image
        img = img.resize(size, Image.ANTIALIAS)

        # Assuming that the image is in RGBA format and the background is white
        img = img.convert("RGBA")
        data = img.getdata()

        new_data = []
        for item in data:
            # Change all white (also shades of whites)
            # (also shades of grey if you want) to transparent
            if item[0] in list(range(200, 256)):
                new_data.append((255, 255, 255))
            else:
                new_data.append(item)

        img.putdata(new_data)

        # Crop the image to the area containing non-transparent pixels
        img = img.crop(img.getbbox())

        # Save the image
        img.save(output_path)


def create_subsets(X, y, num_subsets):
    """
    Divides the features (X) into a specified number of subsets and concatenates each subset with the target (y).

    Parameters
    ----------
    X : pd.DataFrame
        The input features.
    y : pd.Series
        The target values.
    num_subsets : int
        The number of subsets to divide X into.

    Returns
    -------
    subsets : list of pd.DataFrame
        A list of dataframes, each consisting of a subset of features and the target.
    """

    # Calculate number of columns per subset
    cols_per_subset = X.shape[1] // num_subsets

    # Create list to store subsets
    subsets = []

    for i in range(num_subsets):
        if i == num_subsets - 1:  # for the last subset, add all remaining columns
            subset = pd.concat([X.iloc[:, i * cols_per_subset :], y], axis=1)
        else:
            subset = pd.concat(
                [X.iloc[:, i * cols_per_subset : (i + 1) * cols_per_subset], y], axis=1
            )
        subsets.append(subset)

    return subsets


def plot_heatmaps(subsets):
    """
    Plots and saves correlation heatmaps for each subset of data.

    Parameters
    ----------
    subsets : list of pd.DataFrame
        A list of dataframes for which to plot the correlation heatmaps.
    """

    for i, subset in enumerate(subsets):
        corr = subset.corr()

        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values,
            showscale=True,
            colorscale="plotly3",
        )

        fig.update_layout(
            title=f"Correlation Heatmap of Subset {i+1}",
            height=1200,
            width=1200,
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.show()
        fig.write_image(f"heatmaps/heatmap_subset_{i+1}.png", scale=2)


def group_by_model(df):
    """
    Groups the data by model and calculates the cumulative sum of time consumption for each model.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing model information.

    Returns
    -------
    df_list : list of pd.DataFrame
        A list of dataframes, each containing the information for one model.
    """

    df_list = []
    model_groups = df.groupby("model")

    for name, group in model_groups:
        group = group.copy()  # to avoid SettingWithCopyWarning
        group["total_time_consumption"] = group["time_consumption"].cumsum()
        df_list.append(group)

    return df_list


def format_number(value):
    """
    Formats a number to a string with suffixes (B for billion, M for million, k for thousand).

    Parameters
    ----------
    value : int or float
        The number to format.

    Returns
    -------
    str
        The formatted number.
    """

    if abs(value) >= 1e9:
        return "{:.1f}B".format(value / 1e9)
    elif abs(value) >= 1e6:
        return "{:.1f}M".format(value / 1e6)
    elif abs(value) >= 1000:
        return "{:.1f}k".format(value / 1000)
    else:
        return "{:.1f}".format(value)


def save_keras_models(best_models):
    """
    Saves the diagrams of best Keras models in PNG format.

    Parameters
    ----------
    best_models : list of dict
        A list of dictionaries, each containing a Keras model and other related information.
    """

    # Create a directory to store the images
    os.makedirs("model_images", exist_ok=True)

    for model_info in best_models:
        model = model_info["model"]

        # Check if the model is a Keras model
        if hasattr(model, "summary"):
            # Generate the image filename
            filename = (
                f'model_images/best_models/model_{model_info["step"]}_{model.name}.png'
            )

            # Create the diagram
            plot_model(model, to_file=filename, show_shapes=True)
        else:
            print(
                f"Model at step {model_info['step']} is not a Keras model and can't be visualized."
            )


def save_dataframe(df_list, image_dir="model_images/all_models/"):
    """
    Saves the dataframes as images in the specified directory.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        A list of dataframes to be saved as images.
    image_dir : str, optional
        The directory to save the images. Default is 'model_images/all_models/'.
    """

    os.makedirs(image_dir, exist_ok=True)

    # Mapping of old column names to new column names
    column_name_mapping = {
        "model": "Model",
        "step": "Step",
        "val_metric": "Validation Metric (MSE)",
        "time_consumption": "Time",
        "total_time_consumption": "Total Time",
    }

    # Mapping of old model names to new model names
    model_name_mapping = {
        "AdaBoostRegressorModel": "Ada Boost",
        "BaggingRegressorModel": "Bagging",
        "BayesianRidgeRegressorModel": "Bayesian Ridge",
        "CnnRegressorModel": "CNN",
        "DecisionTreeRegressorModel": "Decision Tree",
        "ElasticNetRegressorModel": "Elastic Net",
        "ExtraTreesRegressorModel": "Extra Trees",
        "GaussianProcessRegressorModel": "Gaussian Process",
        "GradientBoostingRegressorModel": "Gradient Boosting",
        "KNNRegressorModel": "k-Nearest Neighbors",
        "LassoRegressorModel": "Lasso",
        "LightGradientBoostingRegressorModel": "LGBM",
        "LstmRegressorModel": "LSTM",
        "NnRegressorModel": "Neural Network",
        "NuSVRRegressorModel": "NuSVR",
        "PassiveAggressiveRegressorModel": "Passive Aggressive",
        "RandomForestRegressorModel": "Random Forest",
        "RidgeRegressorModel": "Ridge",
        "SupportVectorRegressorModel": "Support Vector",
        "TweedieRegressorModel": "Tweedie",
        "eXtremeGradientBoostingRegressorModel": "XGB",
    }

    for model_df in df_list:
        # Rename column names
        model_df.rename(columns=column_name_mapping, inplace=True)

        # Rename model names
        model_df["Model"] = model_df["Model"].replace(model_name_mapping)

        # Rename column names
        model_df.rename(columns=column_name_mapping, inplace=True)

        # Define the default widths for each column here, in pixels.
        column_widths = [100] * len(model_df.columns)

        # Modify the widths for specific columns
        column_widths[model_df.columns.get_loc("Model")] = 300
        column_widths[model_df.columns.get_loc("Total Time")] = 200
        column_widths[model_df.columns.get_loc("Step")] = 50
        column_widths[model_df.columns.get_loc("Time")] = 50

        # Adjust the width based on the number of columns (assuming 150 pixels per column)
        width = sum(column_widths) + 50  # added 50 to account for padding

        # Format the float columns to desired precision
        float_cols = ["Time", "Total Time"]

        formatted_cols = []
        for col in model_df.columns:
            if col == "Validation Metric (MSE)":
                formatted_cols.append(model_df[col].map(format_number))
            elif col in float_cols:
                # Convert time in seconds to minutes
                formatted_cols.append(
                    model_df[col].map(lambda x: "{:.2f} min".format(x / 60))
                )
            else:
                formatted_cols.append(model_df[col])

        fig = go.Figure(
            data=[
                go.Table(
                    columnwidth=column_widths,
                    header=dict(
                        values=list(model_df.columns),
                        fill_color="paleturquoise",
                        align="center",
                    ),
                    cells=dict(
                        values=formatted_cols,
                        fill_color="lavender",
                        align="center",
                    ),
                )
            ]
        )

        fig.update_layout(
            width=width,
            height=1000,
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Use the model's name as the filename
        model_name = model_df["Model"].iloc[0]
        lowercase_model_name = model_name.lower().replace(" ", "_")
        fig.write_image(f"{image_dir}{lowercase_model_name}.png")


def read_model_metrics(log_filename):
    """
    Reads a log file and extracts model metrics into a pandas DataFrame.

    Parameters
    ----------
    log_filename : str
        The path to the log file.

    Returns
    -------
    df_sorted : pd.DataFrame
        A dataframe containing the extracted model metrics, sorted by model and step.
    """

    metrics = []
    with open(log_filename, "r") as file:
        for line in file.readlines():
            match = re.search(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - Model: (\w+), Step: (\d+), Val Metric: (\d+.\d+), Time Consumption: (\d+.\d+)",
                line,
            )
            if match:
                metric = {
                    "model": match.group(2),
                    "step": int(match.group(3)),
                    "val_metric": float(match.group(4)),
                    "time_consumption": float(match.group(5)),
                }
                metrics.append(metric)
    df = pd.DataFrame(metrics)

    # Sorting by model and step
    df_sorted = df.sort_values(["model", "step"])

    return df_sorted


def save_metrics(df, image_dir="model_images/metrics/"):
    """
    Saves each model's metrics as a separate PNG image in the specified directory.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing model information.
    image_dir : str, optional
        The directory to save the images. Default is 'model_images/metrics/'."""

    # Create directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)

    # Visualize data
    models = df["model"].unique()
    for model in models:
        model_df = df[df["model"] == model]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=model_df["step"], y=model_df["val_metric"], mode="lines", name=model
            )
        )

        fig.update_layout(
            title=f"Metrics for {model}",
            xaxis_title="Step",
            yaxis_title="Validation Metric",
            autosize=True,
        )

        fig.write_image(
            os.path.join(image_dir, f"{model}_metrics.png")
        )  # Save each model's metrics as a separate PNG


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between the true and predicted values.

    Parameters
    ----------
        y_true (np.ndarray): The actual values.
        y_pred (np.ndarray): The predicted values.

    Returns
    -------
        float: The MAPE.
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_mean_squared_log_error(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error (RMSLE) between the true and predicted values.

    Parameters
    ----------
        y_true (np.ndarray): The actual values.
        y_pred (np.ndarray): The predicted values.

    Returns
    -------
        float: The RMSLE.
    """

    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def plot_price_prediction(X_test, y_test, predictions, title):
    """
    Plots the predicted and actual values for the test data.

    Parameters
    ----------
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The actual values for the test data.
        predictions (pd.Series): The predicted values for the test data.
        title (str): The title of the plot.

    Returns
    -------
        plotly.graph_objects._figure.Figure: The figure object of the plot.
    """

    # Ensure dates are sorted in ascending order
    X_test = X_test.sort_index(ascending=True)
    y_test = y_test.sort_index(ascending=True)
    predictions = predictions.sort_index(ascending=True)

    if len(predictions.columns) > 3:
        rows = len(predictions.columns) // 3 + (len(predictions.columns) % 3 > 0)
        fig = make_subplots(rows=rows, cols=3, subplot_titles=predictions.columns)

        for i, col in enumerate(predictions.columns, start=1):
            row = i // 3 + (i % 3 > 0)
            col_pos = i % 3 if i % 3 != 0 else 3
            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions[col],
                    mode="lines",
                    name="Predicted " + col,
                ),
                row=row,
                col=col_pos,
            )
            # Plot actual test value for each subplot
            for y_col in y_test.columns:
                fig.add_trace(
                    go.Scatter(
                        x=y_test.index,
                        y=y_test[y_col],
                        mode="lines",
                        name="Actual " + y_col,
                    ),
                    row=row,
                    col=col_pos,
                )

        fig.update_layout(
            title_text=title,
            showlegend=True,
            xaxis_title="Date",
            yaxis_title="Price",
            height=300 * rows,
            width=1600,
        )

    else:
        fig = go.Figure()
        # Plot each prediction
        for col in predictions.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions.index,
                    y=predictions[col],
                    mode="lines",
                    name="Predicted " + col,
                )
            )

        # Plot each actual test value
        for col in y_test.columns:
            fig.add_trace(
                go.Scatter(
                    x=y_test.index, y=y_test[col], mode="lines", name="Actual " + col
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend",
            height=500,
            width=1600,
        )

    return fig


def plot_histories(histories):
    """
    Plots the loss histories for different training epochs.

    Parameters
    ----------
        histories (list of keras.callbacks.History): The list of history objects for each training epoch.
    """

    subplot_titles = [f"History at Window {i+1}" for i in range(len(histories))]

    rows = len(histories) // 4 if len(histories) % 4 == 0 else len(histories) // 4 + 1
    fig = make_subplots(rows=rows, cols=4, subplot_titles=subplot_titles)
    row_counter = 1
    col_counter = 1

    for i, history in enumerate(histories):
        fig.add_trace(
            go.Scatter(y=history.history["loss"], mode="lines", name="Training loss"),
            row=row_counter,
            col=col_counter,
        )
        fig.add_trace(
            go.Scatter(
                y=history.history["val_loss"], mode="lines", name="Validation loss"
            ),
            row=row_counter,
            col=col_counter,
        )

        last_epoch = len(history.history["loss"])
        training_loss = history.history["loss"][-1]
        validation_loss = history.history["val_loss"][-1]

        fig.add_annotation(
            x=last_epoch,
            y=training_loss,
            xref="x",
            yref="y",
            text=f"Training loss: {training_loss:.4f}",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40,
            font=dict(color="blue"),
            row=row_counter,
            col=col_counter,
        )

        fig.add_annotation(
            x=last_epoch,
            y=validation_loss,
            xref="x",
            yref="y",
            text=f"Validation loss: {validation_loss:.4f}",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40,
            font=dict(color="orange"),
            row=row_counter,
            col=col_counter,
        )

        col_counter += 1
        if col_counter > 4:
            col_counter = 1
            row_counter += 1

    fig.update_layout(height=400 * rows, width=1600, title_text="Model losses")
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    fig.show()


def reverse_values(predictions, X_scaled, y_scaled, scaler):
    """
    Reverses the effect of scaling on the predictions and the scaled features and target.

    Parameters
    ----------
        predictions (pd.DataFrame): The predicted values.
        X_scaled (pd.DataFrame): The scaled features.
        y_scaled (pd.Series): The scaled target.
        scaler (sklearn.preprocessing.StandardScaler): The scaler used to scale the data.

    Returns
    -------
        reverse_predictions_df (pd.DataFrame): The unscaled predictions.
        reverse_x_df (pd.DataFrame): The unscaled features.
        reverse_y_df (pd.Series): The unscaled target.
    """

    reverse_predictions_df = pd.DataFrame(index=predictions.index)

    for column in predictions.columns:
        reverse_predictions = scaler.inverse_transform(
            predictions[column].values.reshape(-1, 1)
        )
        reverse_predictions_df[column] = reverse_predictions.ravel()

    reverse_x = scaler.inverse_transform(X_scaled.values)
    reverse_x_df = pd.DataFrame(
        reverse_x, columns=X_scaled.columns, index=X_scaled.index
    )

    reverse_y = scaler.inverse_transform(y_scaled.values)
    reverse_y_df = pd.DataFrame(
        reverse_y, columns=y_scaled.columns, index=y_scaled.index
    )

    return reverse_predictions_df, reverse_x_df, reverse_y_df


def find_best_models(metric_data):
    """
    Finds the best models based on their validation metric.

    Parameters
    ----------
        metric_data (list of dict): A list of dictionaries, each containing the information for one model.

    Returns
    -------
        best_models (list of dict): A list of dictionaries, each containing the information for one best model.
    """

    best_models = []

    best_val_metrics = {}

    for data in metric_data:
        step = data["step"]
        model = data["model"]
        val_metric = data["val_metric"]

        if step not in best_val_metrics or val_metric < best_val_metrics[step]:
            best_val_metrics[step] = val_metric
            best_model = {"step": step, "model": model, "val_metric": val_metric}

            existing_steps = [model["step"] for model in best_models]
            if step in existing_steps:
                index = existing_steps.index(step)
                best_models[index] = best_model
            else:
                best_models.append(best_model)

    return best_models
