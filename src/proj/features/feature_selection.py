
from proj.evaluation.backtesting import rolling_forecast_backtest, evaluate_performance, ts_cv_score



import pandas as pd

from proj.evaluation.backtesting import rolling_forecast_backtest, evaluate_performance, ts_cv_score
import pandas as pd


def greedy_feature_selection(model_cls, model_params, X, y, returns, feature_lim=3, train_size=0.8, horizon=1, verbose=True):
    """
    Greedy forward feature selection based on the 'qlike' metric (lower is better),
    compatible with the new GARCHRegressor API (returns passed to fit()).

    Parameters
    ----------
    model : callable / class
        Model class (e.g., GARCHRegressor).
    model_params : dict
        Hyperparameters for the model ONLY (no data like returns).
    X : pd.DataFrame
        Full feature matrix (candidate exogenous regressors).
    y : pd.Series or np.array
        Target time series (e.g., RV / logRV) used for scoring.
    returns : pd.Series or np.array
        Return series aligned 1:1 with y (used to fit GARCH)
    feature_lim : int
        Maximum number of features to select (default 3). If None/0 -> all.
    train_size : float
        Initial training fraction for expanding-window splits.
    horizon : int
        Forecast horizon.
    verbose : bool
        Print progress.

    Returns
    -------
    selected_features : list[str]
        Selected feature names in the order added.
    history_df : pd.DataFrame
        Rows: step, added_feature, qlike
    """
    if not hasattr(X, "columns"):
        raise ValueError("X must be a pandas DataFrame with named columns.")

    params_left = list(X.columns)
    selected_features = []
    history = []

    # ---- Baseline: no exogenous features ----
    base_results = rolling_forecast_backtest(
        model_cls=model,
        model_params=model_params,
        y=y,
        X=None,
        returns=returns,
        train_size=train_size,
        horizon=horizon,
    )
    base_metrics = evaluate_performance(base_results)
    current_qlike = base_metrics["qlike"]

    history.append({"step": 0, "added_feature": "base", "qlike": current_qlike})
    if verbose:
        print(f"Initial qlike (no features): {current_qlike}")

    if not feature_lim:
        feature_lim = len(params_left)

    # ---- Greedy forward selection ----
    for _ in range(feature_lim):
        best_col = None
        best_qlike = current_qlike

        for col in params_left:
            candidate_features = selected_features + [col]
            X_candidate = X[candidate_features]

            qlike_x = ts_cv_score(
                model_cls=model,
                params=model_params,
                y=y,
                X=X_candidate,
                returns=returns,
                train_size=train_size,
                horizon=horizon,
            )

            if qlike_x < best_qlike:
                best_qlike = qlike_x
                best_col = col

        if best_col is None:
            if verbose:
                print("No further improvement from adding any feature.")
            break

        selected_features.append(best_col)
        params_left.remove(best_col)
        current_qlike = best_qlike

        history.append(
            {"step": len(selected_features), "added_feature": best_col, "qlike": current_qlike}
        )

        if verbose:
            print(f"Step {len(selected_features)}: added {best_col}, qlike = {current_qlike}")

    history_df = pd.DataFrame(history)
    return selected_features, history_df
