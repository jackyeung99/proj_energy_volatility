from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def build_regime_table(
    fit: Dict[str, Any],
    run_id: str,
) -> pd.DataFrame:
    """
    Build daily regime table with:
      p_state_i, p_state_i_next, labels, confidence
    """
    y: pd.Series = fit["y"]
    filtered: pd.DataFrame = fit["filtered"]
    P: np.ndarray = fit["P"]
    k_states = filtered.shape[1]

    # Next-step probs for each date: (T, K)
    next_vals = filtered.to_numpy(dtype=float) @ P
    next_df = pd.DataFrame(
        next_vals,
        index=filtered.index,
        columns=[f"p_state_{i}_next" for i in range(k_states)],
    )

    # Stable labels by mean(y) under posterior weights
    state_map = label_states_by_mean(y, filtered)

    state_id = (
        filtered.idxmax(axis=1)
        .str.replace("p_state_", "", regex=False)
        .astype(int)
    )
    state_label = state_id.map(state_map)

    next_state_id = (
        next_df.idxmax(axis=1)
        .str.replace("p_state_", "", regex=False)
        .str.replace("_next", "", regex=False)
        .astype(int)
    )
    next_state_label = next_state_id.map(state_map)

    confidence = filtered.max(axis=1).astype(float)
    confidence_next = next_df.max(axis=1).astype(float)

    out = pd.concat([filtered, next_df], axis=1)

    pi = out[[f"p_state_{i}" for i in range(k_states)]].to_numpy()
    stay_prob = (pi * np.diag(P)).sum(axis=1)
    out["switch_risk_next"] = 1.0 - stay_prob
    out["stay_prob_next"] = stay_prob


    out["state_id"] = state_id.to_numpy()
    out["state_label"] = state_label.to_numpy()
    out["confidence"] = confidence.to_numpy()

    out["state_id_next"] = next_state_id.to_numpy()
    out["state_label_next"] = next_state_label.to_numpy()
    out["confidence_next"] = confidence_next.to_numpy()

    # metadata for debugging / reproducibility
    out["run_id"] = run_id
    out["k_states"] = int(k_states)

    return out

def fit_markov_regime_model(
    y: pd.Series,
    k_states: int = 3,
    switching_variance: bool = False,
    trend: str = "c",
    em_iter: int = 10,
    maxiter: int = 200,
) -> Dict[str, Any]:
    """
    Fit MarkovRegression and return normalized objects needed to build a regime table.
    """
    if not isinstance(y.index, pd.DatetimeIndex):
        raise ValueError("y must have a DatetimeIndex")
    y = y.sort_index().astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(y) < 100:
        raise ValueError(f"Need more data to fit Markov model. Have {len(y)} points, want 100+.")

    mod = MarkovRegression(
        endog=y,
        k_regimes=k_states,
        trend=trend,
        switching_variance=switching_variance,
    )
    res = mod.fit(em_iter=em_iter, maxiter=maxiter, disp=False)

    # Filtered marginal probabilities (T, K)
    filtered = res.filtered_marginal_probabilities.copy()
    # Ensure columns are consistent
    if isinstance(filtered, pd.DataFrame):
        filtered.columns = [f"p_state_{i}" for i in range(k_states)]
    else:
        # If some version returns ndarray
        filtered = pd.DataFrame(
            np.asarray(filtered, dtype=float),
            index=y.index,
            columns=[f"p_state_{i}" for i in range(k_states)],
        )

    P = _get_transition_matrix(res, k_states)

    return {
        "res": res,
        "y": y,
        "filtered": filtered,
        "P": P,
    }


def _get_transition_matrix(res, k_states: int) -> np.ndarray:
    """
    Return a (K, K) transition matrix robustly across statsmodels versions.
    """
    # Prefer the constant transition_matrix if present
    P = getattr(res, "transition_matrix", None)
    if P is not None:
        P = np.asarray(P, dtype=float)
        if P.shape == (k_states, k_states):
            return P

    # Fall back to regime_transition but squeeze down to 2D if needed
    P = np.asarray(getattr(res, "regime_transition"), dtype=float)

    # common nuisance shapes: (K, K, 1) or (K, K, T)
    if P.ndim == 3:
        # If time-varying, last slice is a reasonable default for "current" transition
        # and is consistent with using last filtered probs for next-step.
        P = P[:, :, -1]

    if P.ndim != 2 or P.shape != (k_states, k_states):
        raise ValueError(f"Transition matrix has shape {P.shape}, expected {(k_states, k_states)}")

    return P

def label_states_by_mean(y: pd.Series, filtered_probs: pd.DataFrame) -> dict[int, str]:
    """
    Optional helper: label state ids by level (Low/Med/High) based on implied mean of y.
    This gives stable human-readable labels.

    Returns mapping: {state_id: "Low"/"Medium"/"High"/...}
    """
    # y = _as_1d_series(y)
    k = filtered_probs.shape[1]

    # "Soft" mean estimate per state using posterior weights
    means = []
    for i in range(k):
        w = filtered_probs.iloc[:, i].reindex(y.index).fillna(0.0).to_numpy()
        mu = np.sum(w * y.to_numpy()) / max(np.sum(w), 1e-12)
        means.append(mu)

    order = np.argsort(means)  # low mean -> high mean
    if k == 2:
        names = ["Low", "High"]
    elif k == 3:
        names = ["Low", "Medium", "High"]
    else:
        names = [f"StateRank{i}" for i in range(k)]

    return {int(state): names[rank] for rank, state in enumerate(order)}

