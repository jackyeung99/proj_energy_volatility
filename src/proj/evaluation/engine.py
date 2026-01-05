import pandas as pd
import numpy
from typing import Callable, Dict, List, Optional, Tuple, Any



def walkforward_points(n: int, initial_train: int, step: int = 1):
    """Yield (train_end, test_idx) where test is a single point."""
    if initial_train < 1 or initial_train >= n:
        raise ValueError("initial_train must be in [1, n-1]")
    for t in range(initial_train, n, step):
        yield t, t  # train is [0:t), test is [t]

def evaluate_models(bt, models):
    """
    Run backtest for each model and compute:
      - qlike (bt.metric)
      - corr(truth, forecast)
    Returns a summary DataFrame.
    """
    rows = []

    for model in models:
        res = bt.run(model, store=False)

        y = res["truth"].astype(float).to_numpy()
        f = res["forecast"].astype(float).to_numpy()

        # qlike (vectorized metric)
        q = float(bt.metric(y, f))

        # correlation (guard against NaNs / constants)
        mask = np.isfinite(y) & np.isfinite(f)
        if mask.sum() >= 2 and np.std(y[mask]) > 0 and np.std(f[mask]) > 0:
            corr = float(np.corrcoef(y[mask], f[mask])[0, 1])
        else:
            corr = np.nan

        rows.append({
            "model": model.name,
            "qlike": q,
            "corr": corr,
            "n": int(mask.sum()),
        })

    return pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)

d
def model_selection(backtester, models):
    """
    Score a list of models using the backtester.

    Parameters
    ----------
    backtester : Backtester
    models : iterable
        Iterable of *model instances* (already constructed)

    Returns
    -------
    dict[str, float]
        Mapping model.name -> score
    """
    scores = {}
    for model in models:
        score = backtester.score_model(model)
        scores[model.name] = score
        print(f"{model.name}: {score:.6f}")
    return scores





def greedy_forward_feature_selection(
    backtester,
    model_factory: Callable[[Dict[str, Any], List[str]], Any],
    param_fixed: Optional[Dict[str, Any]],
    candidate_features: List[str],
    feature_lim: int = 3,
    tol: float = 0.0,
    verbose: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Greedy forward feature selection using Backtester as the scoring authority.

    Parameters
    ----------
    backtester : Backtester
        Your evaluation/backtesting object with .score_model(model)->float.
    model_factory : callable
        Function: (params_dict, x_cols_list) -> NEW model instance.
        Example:
            lambda params, x_cols: HARRV(x_cols=x_cols, **params)
    param_fixed : dict
        Hyperparameters held fixed during feature selection (may be {}).
    candidate_features : list[str]
        Pool of features to choose from.
    feature_lim : int
        Max number of features to select.
    tol : float
        Require improvement of at least tol to accept a new feature.
    verbose : bool

    Returns
    -------
    selected : list[str]
    history : pd.DataFrame with columns: step, added_feature, score
    """
    param_fixed = {} if param_fixed is None else dict(param_fixed)

    remaining = list(candidate_features)
    selected: List[str] = []
    history_rows = []

    # ---- baseline: no features
    base_model = model_factory(param_fixed, [])
    best_score = backtester.score_model(base_model)

    history_rows.append({"step": 0, "added_feature": "base", "score": best_score})
    if verbose:
        print(f"Baseline score (no features): {best_score:.6f}")

    if feature_lim is None or feature_lim <= 0:
        feature_lim = len(remaining)

    # ---- greedy add
    for step in range(1, feature_lim + 1):
        best_feat = None
        best_step_score = best_score

        for feat in remaining:
            trial_feats = selected + [feat]
            trial_model = model_factory(param_fixed, trial_feats)
            s = backtester.score_model(trial_model)

            if verbose:
                print(f"  try +{feat}: score={s:.6f}")

            if s + tol < best_step_score:
                best_step_score = s
                best_feat = feat

        if best_feat is None:
            if verbose:
                print("No further improvement from adding any feature.")
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_score = best_step_score

        history_rows.append({"step": step, "added_feature": best_feat, "score": best_score})
        if verbose:
            print(f"✅ Step {step}: added {best_feat}, score={best_score:.6f}")

    return selected, pd.DataFrame(history_rows)


def greedy_hyperparam_tuning(
    backtester,
    model_factory: Callable[[Dict[str, Any], List[str]], Any],
    param_grid: Dict[str, List[Any]],
    x_cols: Optional[List[str]] = None,
    start_params: Optional[Dict[str, Any]] = None,
    param_order: Optional[List[str]] = None,
    max_passes: int = 5,
    tol: float = 1e-8,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
    """
    Greedy coordinate-descent hyperparameter tuning using Backtester scoring.

    Parameters
    ----------
    backtester : Backtester
        Must provide score_model(model)->float.
    model_factory : callable
        Function: (params_dict, x_cols_list) -> NEW model instance.
    param_grid : dict[str, list]
        Hyperparameter candidates.
    x_cols : list[str]
        Fixed feature set during hyperparam tuning.
    start_params : dict
        Starting point (optional). Missing keys default to first grid value.
    param_order : list[str]
        Order to tune parameters (optional).
    max_passes : int
    tol : float
        Require improvement of at least tol to accept update.
    verbose : bool

    Returns
    -------
    best_params : dict
    best_score : float
    history_df : DataFrame of updates and pass summaries
    """
    if not param_grid:
        raise ValueError("param_grid must be a non-empty dict of lists.")
    x_cols = [] if x_cols is None else list(x_cols)

    keys = list(param_grid.keys())
    if param_order is not None:
        keys = [k for k in param_order if k in param_grid]

    # init params
    current = {}
    if start_params:
        current.update(dict(start_params))
    for k in keys:
        if k not in current:
            vals = list(param_grid[k])
            if not vals:
                raise ValueError(f"param_grid[{k}] is empty.")
            current[k] = vals[0]

    # score initial
    best_model = model_factory(current, x_cols)
    best_score = backtester.score_model(best_model)

    if verbose:
        print("Initial params:", current)
        print(f"Initial score: {best_score:.6f}")

    history = [{
        "type": "init",
        "params": dict(current),
        "score": float(best_score),
    }]

    # coordinate descent passes
    for p in range(1, max_passes + 1):
        improved = False
        if verbose:
            print(f"\n=== Pass {p}/{max_passes} ===")

        for k in keys:
            base_val = current[k]
            best_k_val = base_val
            best_k_score = best_score

            candidates = list(param_grid[k])
            if base_val not in candidates:
                candidates = [base_val] + candidates

            for v in candidates:
                trial = dict(current)
                trial[k] = v
                s = backtester.score_model(model_factory(trial, x_cols))

                if verbose:
                    print(f"  {k}={v!r} -> score={s:.6f}")

                if s + tol < best_k_score:
                    best_k_score = s
                    best_k_val = v

            if best_k_val != base_val and best_k_score + tol < best_score:
                current[k] = best_k_val
                best_score = best_k_score
                improved = True

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
            "improved": bool(improved),
        })

        if not improved:
            if verbose:
                print("\nNo improvement this pass. Stopping.")
            break

    return dict(current), float(best_score), pd.DataFrame(history)
