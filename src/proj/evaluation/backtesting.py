import numpy as np 
from proj.evaluation.metrics import rmse, qlike
from sklearn.model_selection import TimeSeriesSplit


def rolling_forecast_backtest(
    model_cls,
    model_params,
    y,
    X=None,
    train_size=.8,
    horizon=1,
):
    y = np.asarray(y)
    if X is not None:
        X = np.asarray(X)

    n = len(y)
    initial_train_size = int(n * train_size)

    # How many splits? Make each test fold length = horizon
    # and ensure the first training fold is at least initial_train_size
    n_splits = (n - initial_train_size) // horizon

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=horizon,
    )

    preds = []
    trues = []

    split_idx = 0
    for train_idx, test_idx in tscv.split(y):
        # enforce minimum initial train size
        if len(train_idx) < initial_train_size:
            continue

        y_train, y_test = y[train_idx], y[test_idx]
        if X is not None:
            X_train, X_test = X[train_idx], X[test_idx]
        else:
            X_train = X_test = None

        model = model_cls(**model_params)
        model.fit(y_train, X_train)

        # For sklearn-style regressors, we usually just predict on X_test
        if hasattr(model, "predict") and "X_future" not in model.predict.__code__.co_varnames:
            y_pred = model.predict(X_test)
        else:
            # for your custom interface
            y_pred = model.predict(horizon=len(test_idx), X_future=X_test)

        preds.append(y_pred)
        trues.append(y_test)

        split_idx += 1

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return {
        "y_true": trues,
        "y_pred": preds,
    }


def ts_cv_score(model_cls, model_params, y, X=None, n_splits=3, horizon=1, metric=qlike):
    """
    Returns average CV score (lower is better).
    """
    y = np.asarray(y)
    if X is not None:
        X = np.asarray(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    for train_idx, val_idx in tscv.split(y):
        y_train, y_val = y[train_idx], y[val_idx]
        X_train = X[train_idx] if X is not None else None
        X_val   = X[val_idx]   if X is not None else None

        model = model_cls(**model_params)
        model.fit(y_train, X_train)
        y_pred = model.predict(horizon=len(val_idx), X_future=X_val)

        scores.append(metric(y_val, y_pred))

    return np.mean(scores)


def evaluate_performance(results):
    y_true = results['y_true']
    y_pred = results['y_pred']

    return {
        'rmle': rmse(y_true, y_pred),
        'qlike': qlike(y_true, y_pred)
    }




