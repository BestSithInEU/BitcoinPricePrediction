from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_log_error

import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras.utils import plot_model


def group_by_model(df):
    df_list = []
    model_groups = df.groupby("model")

    for name, group in model_groups:
        group = group.copy()  # to avoid SettingWithCopyWarning
        group["total_time_consumption"] = group["time_consumption"].cumsum()
        df_list.append(group)

    return df_list


def format_number(value):
    if abs(value) >= 1e9:
        return "{:.1f}B".format(value / 1e9)
    elif abs(value) >= 1e6:
        return "{:.1f}M".format(value / 1e6)
    elif abs(value) >= 1000:
        return "{:.1f}k".format(value / 1000)
    else:
        return "{:.1f}".format(value)


def save_keras_models(best_models):
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
        "CNNRegressorModel": "CNN",
        "DecisionTreeRegressorModel": "Decision Tree",
        "ElasticNetRegressorModel": "Elastic Net",
        "ExtraTreesRegressorModel": "Extra Trees",
        "GaussianProcessRegressorModel": "Gaussian Process",
        "GradientBoostingRegressorModel": "Gradient Boosting",
        "KNNRegressorModel": "k-Nearest Neighbors",
        "LassoRegressorModel": "Lasso",
        "LightGradientBoostingRegressorModel": "LGBM",
        "LSTMRegressorModel": "LSTM",
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


def visualize_metrics(df, image_dir="model_images/metrics/"):
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig.write_image(
            os.path.join(image_dir, f"{model}_metrics.png")
        )  # Save each model's metrics as a separate PNG


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def plot_price_prediction(X_test, y_test, predictions, title):
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
