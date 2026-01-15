import pandas as pd
import numpy as np 


def compute_rolling_logrv_stats(
    gold: pd.DataFrame,
    realized_col: str,
    eps: float,
    window_bd: int = 60,
) -> pd.DataFrame:
    """
    Rolling stats in log-space, anchored to a rolling moving average baseline.

    Returns:
      - ma_logrv: rolling mean of log(RV)
      - sd_dev_logrv: rolling std of (log(RV) - ma_logrv)
    """
    g = gold.copy().sort_index()
    g.index = pd.to_datetime(g.index, utc=True)

    x = np.log(g[realized_col].astype(float) + eps)

    minp = max(10, window_bd // 3)
    ma = x.rolling(window_bd, min_periods=minp).mean()
    dev = x - ma
    sd_dev = dev.rolling(window_bd, min_periods=minp).std(ddof=0)

    out = pd.DataFrame({"ma_logrv": ma, "sd_dev_logrv": sd_dev}, index=g.index)
    out.index.name = "ts"
    return out

def _require_k_consecutive(x: pd.Series, k: int = 3) -> pd.Series:
    out = x.copy()
    run_id = (x != x.shift(1)).cumsum()
    run_len = x.groupby(run_id).cumcount() + 1
    out[run_len < k] = np.nan
    return out.ffill().fillna(x.iloc[0])

def add_forecasted_regime_from_gold(
    preds: pd.DataFrame,
    gold: pd.DataFrame,
    realized_col: str,
    eps: float = 1e-12,
    window_bd: int = 60,
    low_p: float = 0.30,
    high_p: float = 0.70,
    tolerance_days: int = 10,
) -> pd.DataFrame:
    if realized_col not in gold.columns:
        raise ValueError(f"gold missing '{realized_col}'. Available: {list(gold.columns)}")
    if "predicted_value" not in preds.columns:
        raise ValueError("preds missing 'predicted_value'")

    stats = compute_rolling_logrv_stats(gold, realized_col=realized_col, eps=eps, window_bd=window_bd)

    # ---- preds: normalize ts and REMOVE any previously-attached stat columns ----
    p = preds.copy()
    p = p.reset_index().rename(columns={p.index.name or "index": "ts"})
    p["ts"] = pd.to_datetime(p["ts"], utc=True)
    p = p.sort_values("ts")

    # drop old baseline/stat columns if they exist to prevent _x/_y suffixing
    for c in ["ma_logrv", "sd_dev_logrv", "mu_logrv", "sd_logrv"]:
        if c in p.columns:
            p = p.drop(columns=[c])

    # (optional) also drop previously computed regime columns if rerunning
    for c in ["regime_z_pred", "regime_pct_pred", "regime_pct_pred_100", "regime_pred", "regime_pred_stable"]:
        if c in p.columns:
            p = p.drop(columns=[c])

    # ---- stats: normalize ts ----
    s = stats.copy()
    s.index = pd.to_datetime(s.index, utc=True)
    s.index.name = "ts"
    s = s.reset_index().sort_values("ts")

    # ---- merge: attach latest past stats ----
    p = pd.merge_asof(
        p, s,
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )

    # now columns should exist UNSUFFIXED
    p = p.dropna(subset=["ma_logrv", "sd_dev_logrv"])

    log_pred = np.log(p["predicted_value"].astype(float) + eps)
    denom = p["sd_dev_logrv"].astype(float).clip(lower=1e-12)
    z = (log_pred - p["ma_logrv"]) / denom

    p["regime_z_pred"] = z
    p["regime_pct_pred"] = norm.cdf(z).clip(0.0, 1.0)
    p["regime_pct_pred_100"] = 100.0 * p["regime_pct_pred"]

    p["regime_pred"] = np.select(
        [p["regime_pct_pred"] < low_p, p["regime_pct_pred"] > high_p],
        ["Low", "High"],
        default="Medium",
    )
    p["regime_pred_stable"] = _require_k_consecutive(p["regime_pred"], k=3)

    p = p.set_index("ts")
    p.index.name = "forecast_close_utc"
    return p