
from proj.evaluation.backtesting import rolling_forecast_backtest, evaluate_performance, ts_cv_score



import pandas as pd

def greedy_feature_selection(model, model_params, X, y, feature_lim=3):
    """
    Greedy forward feature selection based on the 'qlike' metric.
    
    Parameters
    ----------
    model : callable / class
        Model to be used in rolling_forecast_backtest.
    model_params : dict
        Hyperparameters for the model.
    all_params : list
        List of candidate feature names (columns of X).
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series or np.array
        Target time series.
    feature_lim : int
        Maximum number of features to select.
        
    Returns
    -------
    selected_features : list
        List of selected feature names in the order they were added.
    history : pd.DataFrame
        One row per step, with qlike and the feature added.
    """
    
    # Make a copy so we don't mutate the original list
    params_left = list(X.columns)
    selected_features = []
    history = []

    # Baseline model: no exogenous features (X_exog = None)
    base_results = rolling_forecast_backtest(model, model_params, y, None, 0.8)
    base_metrics = evaluate_performance(base_results)
    current_qlike = base_metrics['qlike']

    print(f"Initial qlike (no features): {current_qlike}")

    for step in range(feature_lim):
        best_col = None
        best_qlike = current_qlike

        # Try adding each remaining feature on top of currently selected ones
        for col in params_left:
            candidate_features = selected_features + [col]
            X_candidate = X[candidate_features]

            qlike_x = ts_cv_score(model, model_params, y, X_candidate, 0.8)

            # We assume lower qlike is better
            if qlike_x < best_qlike:
                best_qlike = qlike_x
                best_col = col


        # If no feature improves qlike, stop early
        if best_col is None:
            print("No further improvement from adding any feature.")
            break

        # Accept the best feature of this round
        selected_features.append(best_col)
        params_left.remove(best_col)
        current_qlike = best_qlike

        history.append({
            'step': len(selected_features),
            'added_feature': best_col,
            'qlike': current_qlike
        })

        print(f"Step {len(selected_features)}: added {best_col}, qlike = {current_qlike}")

    history_df = pd.DataFrame(history)
    return selected_features, history_df