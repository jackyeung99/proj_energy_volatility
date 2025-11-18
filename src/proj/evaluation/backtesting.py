import numpy as np 
from proj.evaluation.metrics import rmse, qlike
from sklearn.model_selection import TimeSeriesSplit
from itertools import product


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

        # # For sklearn-style regressors, we usually just predict on X_test
        # if hasattr(model, "predict") and "X_future" not in model.predict.__code__.co_varnames:
        #     y_pred = model.predict(X_test)
        # else:
        #     # for your custom interface
        #     y_pred = model.predict(horizon=len(test_idx), X_future=X_test)

        y_pred = model.predict(horizon=len(test_idx), X_test=X_test)

        preds.append(y_pred)
        trues.append(y_test)

        split_idx += 1

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return {
        "y_true": trues,
        "y_pred": preds,
    }


def ts_cv_score(
    model_cls,
    model_params,
    y,
    X=None,
    train_size=.8,
    horizon=1, 
):
    """
    Returns average CV score (lower is better).
    """
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
    for train_idx, val_idx in tscv.split(y):
        y_train, y_val = y[train_idx], y[val_idx]
        X_train = X[train_idx] if X is not None else None
        X_val   = X[val_idx]   if X is not None else None

        model = model_cls(**model_params)
        model.fit(y_train, X_train)
        y_pred = model.predict(horizon=len(val_idx), X_test=X_val)

        preds.append(y_pred)
        trues.append(y_val)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return qlike(trues, preds)


def ts_hyperparam_search(
    model_cls,
    y,
    X=None,
    param_grid=None,
    train_size=0.8,
    horizon=1,
    verbose=True,
):
    """
    Generic time-series hyperparameter search using ts_cv_score.
    
    Parameters
    ----------
    model_cls : class
        Class of the model, e.g. XGBRVModel or GARCHRegressor.
    y : array-like
        Target series.
    X : array-like or None
        Feature matrix (optional).
    param_grid : dict
        Dictionary of parameter lists, e.g.
        {
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 500],
        }
    train_size : float
        Fraction of data used in the initial training window.
    horizon : int
        Forecast horizon per fold.
    verbose : bool
        If True, print progress.

    Returns
    -------
    best_params : dict
        Parameter combination with lowest CV score.
    best_score : float
        Corresponding score.
    scores : list of (params, score)
        All tried combos and their scores.
    """
    if param_grid is None or len(param_grid) == 0:
        raise ValueError("param_grid must be a non-empty dict of parameter lists.")

    # build all combinations from the grid
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = [dict(zip(keys, v)) for v in product(*values)]

    best_score = np.inf
    best_params = None
    scores = []

    for i, params in enumerate(combos, start=1):
        score = ts_cv_score(
            model_cls=model_cls,
            model_params=params,
            y=y,
            X=X,
            train_size=train_size,
            horizon=horizon,
        )
        scores.append((params, score))

        if verbose:
            print(f"[{i}/{len(combos)}] params={params}, score={score:.6f}")

        if score < best_score:
            best_score = score
            best_params = params

    if verbose:
        print("\nBest params:", best_params)
        print("Best CV score:", best_score)

    return best_params, best_score, scores


def evaluate_performance(results):
    y_true = results['y_true']
    y_pred = results['y_pred']

    return {
        'rmle': rmse(y_true, y_pred),
        'qlike': qlike(y_true, y_pred)
    }




