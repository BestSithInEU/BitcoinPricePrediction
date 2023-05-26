from .save_load_model import save_worker_model
from .rolling_window_trainer import RollingWindowTrainer


def train_model(
    worker_id,
    model_class,
    params,
    X_train_val,
    y_train_val,
    save_dir,
    lstm_stop_early=None,
    nn_stop_early=None,
    cnn_stop_early=None,
):
    model = model_class(**params)

    trainer = RollingWindowTrainer(
        model=model,
        lstm_stop_early=lstm_stop_early,
        nn_stop_early=nn_stop_early,
        cnn_stop_early=cnn_stop_early,
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        train_window=500,
        val_window=100,
        step_size=100,
    )

    trainer.start_training()

    best_model = trainer.best_model
    val_metric = trainer.val_metric
    time_consumption = trainer.time_consumption
    histories = trainer.get_histories()

    result = {
        "best_model": best_model,
        "val_metric": val_metric,
        "time_consumption": time_consumption,
        "histories": histories,
    }

    # Save the trained model
    save_worker_model(worker_id, best_model, save_dir)

    # Free memory by deleting the model object
    del model

    return worker_id, result
