import numpy as np 
from proj.evaluation.metrics import rmse

def rolling_forecast_backtest(
    model_cls,
    model_params,
    y,
    X=None,
    initial_train_size=200,
    horizon=10,
    step=10,
):
    """
    model_cls: class (e.g., ARIMAModel)
    model_params: dict of kwargs for the model
    y: 1D array-like time series
    X: optional features aligned with y
    """
    y = np.asarray(y)
    if X is not None:
        X = np.asarray(X)

    preds = []
    trues = []

    for start in range(initial_train_size, len(y) - horizon + 1, step):
        end_train = start
        end_test = start + horizon

        y_train = y[:end_train]
        y_test = y[end_train:end_test]

        X_train = X[:end_train] if X is not None else None
        X_future = X[end_train:end_test] if X is not None else None

        model = model_cls(**model_params)
        model.fit(y_train, X_train)
        y_pred = model.predict(horizon=horizon, X_future=X_future)

        preds.append(y_pred)
        trues.append(y_test)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return {
        "rmse": rmse(trues, preds),
        "y_true": trues,
        "y_pred": preds,
    }