import numpy as np
from itertools import product
from proj.evaluation.metrics import rmse, qlike


# ----------------------------
# Core utilities
# ----------------------------

def _as_arrays(y, X=None, returns=None):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1D (n,).")

    X = None if X is None else np.asarray(X)
    if X is not None:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

    r = None if returns is None else np.asarray(returns)
    if r is not None:
        if r.ndim != 1:
            r = r.reshape(-1)
        if len(r) != len(y):
            raise ValueError("returns and y must have the same length (aligned).")

    return y, X, r


def _walkforward_splits(n, train_size=0.8, horizon=1):
    """Expanding window walk-forward splits with test blocks of length `horizon`."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    initial_train = int(n * train_size)
    if initial_train <= 0 or initial_train >= n:
        raise ValueError("train_size must leave at least 1 observation for testing.")

    start = initial_train
    while start + horizon <= n:
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, start + horizon)
        yield train_idx, test_idx
        start += horizon


def _fit_predict(model_cls, init_kwargs, y_train, X_train, X_test, returns_train, horizon):
    """
    Fit a model on the training slice and forecast `horizon` steps.
    For ARX mean models, X_test must be shape (horizon, k_exog). For horizon=1,
    (1, k_exog) is fine.
    """
    model = model_cls(**init_kwargs)
    model.fit(y_train, X_train, returns=returns_train)
    yhat = model.predict(X_test, horizon=horizon)  # returns shape (horizon,)
    return np.asarray(yhat).reshape(-1)


def _concat_blocks(blocks):
    """Flatten list of arrays into a single 1D array."""
    if len(blocks) == 0:
        return np.array([], dtype=float)
    blocks = [np.asarray(b).reshape(-1) for b in blocks]
    return np.concatenate(blocks, axis=0)


# ----------------------------
# Backtest / CV
# ----------------------------

def rolling_forecast_backtest(
    model_cls,
    model_params,
    y,
    X=None,
    returns=None,
    train_size=0.8,
    horizon=1,
    model_init_kwargs=None,
):
    """
    Walk-forward backtest.

    IMPORTANT:
    - `returns` must be aligned 1:1 with `y` in time and length.
    - If X is not None and the model uses ARX, each test block passes X_test with
      shape (horizon, k).
    """
    y, X, r = _as_arrays(y, X, returns=returns)
    if r is None:
        raise ValueError("returns is required for GARCHRegressor-style models.")

    model_init_kwargs = {} if model_init_kwargs is None else dict(model_init_kwargs)
    init_kwargs = {**model_init_kwargs, **(model_params or {})}

    preds_blocks, true_blocks = [], []

    for train_idx, test_idx in _walkforward_splits(len(y), train_size, horizon):
        y_train, y_test = y[train_idx], y[test_idx]
        X_train = None if X is None else X[train_idx]
        X_test  = None if X is None else X[test_idx]
        r_train = r[train_idx]  # fit uses returns aligned to training window length

        y_pred = _fit_predict(
            model_cls=model_cls,
            init_kwargs=init_kwargs,
            y_train=y_train,
            X_train=X_train,
            X_test=X_test,
            returns_train=r_train,
            horizon=len(test_idx),  # should equal `horizon`, but safer
        )

        preds_blocks.append(y_pred)
        true_blocks.append(y_test)

    return {
        "y_true_blocks": true_blocks,
        "y_pred_blocks": preds_blocks,
        "y_true": _concat_blocks(true_blocks),
        "y_pred": _concat_blocks(preds_blocks),
    }


def ts_cv_score(
    model_cls,
    params,
    y,
    X=None,
    returns=None,
    train_size=0.8,
    horizon=1,
    model_init_kwargs=None,
    scorer=qlike,
):
    y, X, r = _as_arrays(y, X, returns=returns)
    if r is None:
        raise ValueError("returns is required for GARCHRegressor-style models.")

    model_init_kwargs = {} if model_init_kwargs is None else dict(model_init_kwargs)
    init_kwargs = {**model_init_kwargs, **(params or {})}

    preds_blocks, true_blocks = [], []

    for train_idx, val_idx in _walkforward_splits(len(y), train_size, horizon):
        y_train, y_val = y[train_idx], y[val_idx]
        X_train = None if X is None else X[train_idx]
        X_val   = None if X is None else X[val_idx]
        r_train = r[train_idx]

        y_pred = _fit_predict(
            model_cls=model_cls,
            init_kwargs=init_kwargs,
            y_train=y_train,
            X_train=X_train,
            X_test=X_val,
            returns_train=r_train,
            horizon=len(val_idx),
        )

        preds_blocks.append(y_pred)
        true_blocks.append(y_val)

    y_true = _concat_blocks(true_blocks)
    y_pred = _concat_blocks(preds_blocks)

    return float(scorer(y_true, y_pred))


# ----------------------------
# Hyperparameter search
# ----------------------------

def ts_hyperparam_search(
    model_cls,
    y,
    X=None,
    returns=None,
    param_grid=None,
    train_size=0.8,
    horizon=1,
    verbose=True,
    model_init_kwargs=None,
    scorer=qlike,
    start_params=None,
    param_order=None,
    max_passes=5,
    tol=1e-8,
):
    """
    Greedy / coordinate-descent hyperparameter search for time-series CV.

    Strategy
    --------
    - Initialize params (from start_params or first value of each grid key).
    - For each parameter key:
        - try each candidate value for that key (holding others fixed)
        - keep the best value
    - Repeat passes until no improvement (or max_passes reached).

    Returns
    -------
    best_params : dict
    best_score : float
    history : list[dict]
        One entry per accepted update and per pass summaries.
    """
    if not param_grid:
        raise ValueError("param_grid must be a non-empty dict of parameter lists.")

    model_init_kwargs = {} if model_init_kwargs is None else dict(model_init_kwargs)

    # Determine parameter update order
    keys = list(param_grid.keys())
    if param_order is not None:
        # keep only keys that exist in param_grid, in requested order
        keys = [k for k in param_order if k in param_grid]

    # Initialize parameters
    current = {}
    if start_params:
        current.update(dict(start_params))

    # fill missing keys with first grid value
    for k in keys:
        if k not in current:
            grid_vals = list(param_grid[k])
            if len(grid_vals) == 0:
                raise ValueError(f"param_grid[{k}] is empty.")
            current[k] = grid_vals[0]

    # Score initial
    best_score = ts_cv_score(
        model_cls=model_cls,
        params=current,
        y=y,
        X=X,
        returns=returns,
        train_size=train_size,
        horizon=horizon,
        model_init_kwargs=model_init_kwargs,
        scorer=scorer,
    )

    if verbose:
        print("Initial params:", current)
        print(f"Initial CV score: {best_score:.6f}")

    history = [{
        "type": "init",
        "params": dict(current),
        "score": float(best_score),
    }]

    # Greedy passes
    for p in range(1, max_passes + 1):
        improved_this_pass = False

        if verbose:
            print(f"\n=== Pass {p}/{max_passes} ===")

        for k in keys:
            base_val = current[k]
            best_k_val = base_val
            best_k_score = best_score

            candidates = list(param_grid[k])
            if base_val not in candidates:
                candidates = [base_val] + candidates  # ensure current is considered

            for v in candidates:
                trial = dict(current)
                trial[k] = v

                score = ts_cv_score(
                    model_cls=model_cls,
                    params=trial,
                    y=y,
                    X=X,
                    returns=returns,
                    train_size=train_size,
                    horizon=horizon,
                    model_init_kwargs=model_init_kwargs,
                    scorer=scorer,
                )

                if verbose:
                    print(f"  {k}={v!r} -> score={score:.6f}")

                if score + tol < best_k_score:
                    best_k_score = score
                    best_k_val = v

            # Accept best value for this coordinate if it improves global best
            if best_k_val != base_val and best_k_score + tol < best_score:
                current[k] = best_k_val
                best_score = best_k_score
                improved_this_pass = True

                history.append({
                    "type": "update",
                    "pass": p,
                    "param": k,
                    "value": best_k_val,
                    "params": dict(current),
                    "score": float(best_score),
                })

                if verbose:
                    print(f"  ✅ update: {k}={best_k_val!r} (best score={best_score:.6f})")
            else:
                if verbose:
                    print(f"  (no improvement for {k}; keep {base_val!r})")

        history.append({
            "type": "pass_end",
            "pass": p,
            "params": dict(current),
            "score": float(best_score),
            "improved": bool(improved_this_pass),
        })

        if not improved_this_pass:
            if verbose:
                print("\nNo improvement this pass. Stopping.")
            break

    if verbose:
        print("\nBest params:", current)
        print("Best CV score:", best_score)

    return dict(current), float(best_score), history


# ----------------------------
# Misc
# ----------------------------

def evaluate_performance(results):
    # Prefer flattened arrays for metrics
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    return {"rmse": rmse(y_true, y_pred), "qlike": qlike(y_true, y_pred)}


def model_summary(returns, y_target, X=None, model_params=None, model_cls=None):
    """
    Fit once on the full sample and return (model, arch_result-like object).
    For the new wrapper, fitted result is `model.result_`.
    """
    if model_cls is None:
        raise ValueError("model_cls is required.")
    model_params = {} if model_params is None else dict(model_params)

    y_target = np.asarray(y_target).reshape(-1)
    returns_arr = np.asarray(returns).reshape(-1)
    X_arr = None if X is None else np.asarray(X)

    if X_arr is not None and X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if len(returns_arr) != len(y_target):
        raise ValueError("returns and y_target must be aligned and same length.")

    model = model_cls(**model_params)
    model.fit(y_target, X_arr, returns=returns_arr)

    # consistent with sklearn-style naming
    return model, model.result_
