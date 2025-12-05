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
    model_init_kwargs=None,  
):
    y = np.asarray(y)
    if X is not None:
        X = np.asarray(X)

    if model_init_kwargs is None:
        model_init_kwargs = {}

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

        all_init_kwargs = {**model_init_kwargs, **model_params}
        model = model_cls(**all_init_kwargs)
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
    params,
    y,
    X=None,
    train_size=.8,
    horizon=1,
    model_init_kwargs=None,  
):
    y = np.asarray(y)
    if X is not None:
        X = np.asarray(X)

    if model_init_kwargs is None:
        model_init_kwargs = {}

    n = len(y)
    initial_train_size = int(n * train_size)
    n_splits = (n - initial_train_size) // horizon

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=horizon,
    )

    preds, trues = [], []

    for train_idx, val_idx in tscv.split(y):
        y_train, y_val = y[train_idx], y[val_idx]
        X_train = X[train_idx] if X is not None else None
        X_val   = X[val_idx]   if X is not None else None

        # merge fixed init kwargs + hyperparams
        all_init_kwargs = {**model_init_kwargs, **params}

        model = model_cls(**all_init_kwargs)
        model.fit(y_train, X_train)
        y_pred = model.predict(horizon=len(val_idx), X_test=X_val)

        preds.append(y_pred)
        trues.append(y_val)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return qlike(trues, preds)



def ts_hyperparam_search_full(
    model_cls,
    y,
    X=None,
    param_grid=None,
    train_size=0.8,
    horizon=1,
    verbose=True,
    model_init_kwargs=None,  # NEW
):
    if not param_grid:
        raise ValueError("param_grid must be a non-empty dict of parameter lists.")

    # Build all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = [dict(zip(keys, v)) for v in product(*values)]

    best_score = np.inf
    best_params = None
    scores = []

    for i, params in enumerate(combos, start=1):
        score = ts_cv_score(
            model_cls=model_cls,
            params=params,
            y=y,
            X=X,
            train_size=train_size,
            horizon=horizon,
            model_init_kwargs=model_init_kwargs,
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


def ts_hyperparam_search( 
    model_cls,
    y,
    X=None,
    param_grid=None,
    train_size=0.8,
    horizon=1,
    verbose=True,
    model_init_kwargs=None,
):

    if not param_grid:
        raise ValueError("param_grid must be a non-empty dict of parameter lists.")

    if model_init_kwargs is None:
        model_init_kwargs = {}

    keys = list(param_grid.keys())


    current_params = {k: param_grid[k][0] for k in keys}
    scores = []

    best_score = ts_cv_score(
        model_cls=model_cls,
        params=current_params,
        y=y,
        X=X,
        train_size=train_size,
        horizon=horizon,
        model_init_kwargs=model_init_kwargs,
    )
    scores.append((current_params.copy(), best_score))

    if verbose:
        print("Initial params:", current_params, "score:", best_score)

    eval_counter = 1

    # 2) Coordinate-wise search: optimize each param one at a time
    for k in keys:
        if verbose:
            print(f"\nOptimizing parameter: {k}")

        local_best_score = best_score
        local_best_value = current_params[k]

        for v in param_grid[k]:
            # If this value is the same as current and we already evaluated it,
            # we can skip or re-evaluate. Here we skip to avoid duplicate work.
            if v == current_params[k] and local_best_score == best_score:
                continue

            trial_params = current_params.copy()
            trial_params[k] = v

            score = ts_cv_score(
                model_cls=model_cls,
                params=trial_params,
                y=y,
                X=X,
                train_size=train_size,
                horizon=horizon,
                model_init_kwargs=model_init_kwargs,
            )
            scores.append((trial_params.copy(), score))
            eval_counter += 1

            if verbose:
                print(f"  tried {k}={v}, score={score:.6f}")

            if score < local_best_score:
                local_best_score = score
                local_best_value = v

        # Update current params with the best value found for k
        current_params[k] = local_best_value
        best_score = local_best_score

        if verbose:
            print(f"Best {k} after sweep: {local_best_value}, score={best_score:.6f}")

    if verbose:
        print("\nFinal best params:", current_params)
        print("Final best CV score:", best_score)

    return current_params, best_score, scores


def evaluate_performance(results):
    y_true = results['y_true']
    y_pred = results['y_pred']

    return {
        'rmle': rmse(y_true, y_pred),
        'qlike': qlike(y_true, y_pred)
    }




