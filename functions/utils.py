from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_log_error
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def plot_price_prediction(X_test, y_test, predictions, title):
    fig = go.Figure()

    # Plot each prediction
    for col in predictions.columns:
        fig.add_trace(
            go.Scatter(
                x=X_test.index,
                y=predictions[col],
                mode="lines",
                name="Predicted " + col,
            )
        )

    # Plot each actual test value
    for col in y_test.columns:
        fig.add_trace(
            go.Scatter(
                x=X_test.index, y=y_test[col], mode="lines", name="Actual " + col
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
    )
    return fig


def plot_histories(histories):
    subplot_titles = [f"History at Window {i+1}" for i in range(len(histories))]

    rows = len(histories) // 4 if len(histories) % 4 == 0 else len(histories) // 4 + 1
    fig = sp.make_subplots(rows=rows, cols=4, subplot_titles=subplot_titles)
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


def compare_models(model_metrics, model_names=None):
    if isinstance(model_metrics, dict):
        df = pd.DataFrame(model_metrics)
        df.index.name = "Models"
    elif isinstance(model_metrics, pd.DataFrame):
        df = model_metrics
    else:
        df = pd.DataFrame(model_metrics, index=model_names)
        df.index.name = "Models"

    fig = make_subplots(rows=2, cols=1)

    # MSE comparison
    fig.add_trace(
        go.Bar(x=df.index, y=df["Test MSE"], name="Test MSE", marker_color="indianred"),
        row=1,
        col=1,
    )

    # R2 score comparison
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Test R2 Score"],
            name="Test R2 Score",
            marker_color="lightsalmon",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=600, width=800, title_text="Model Comparison")
    fig.update_yaxes(title_text="Test MSE", row=1, col=1)
    fig.update_yaxes(title_text="Test R2 Score", row=2, col=1)

    fig.show()

    return df


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


def organize_models(all_models, time_consumption, all_val_metric):
    data = {
        "Step": [],
        "Model": [],
        "Time Consumption": [],
        "Validation Metric": [],
        "Model Details": [],
    }

    # Flatten time_consumption and all_val_metric dicts to lists
    time_consumption_list = sum(list(time_consumption.values()), [])
    all_val_metric_list = sum(list(all_val_metric.values()), [])

    for idx, model_info in enumerate(all_models):
        # Here we get the specific model name of the model
        model_class_name = model_info["model"].model_name

        data["Step"].append(model_info["step"])
        data["Model"].append(model_class_name)
        data["Time Consumption"].append(time_consumption_list[idx])
        data["Validation Metric"].append(all_val_metric_list[idx])
        data["Model Details"].append(model_info["model"])

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df


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
