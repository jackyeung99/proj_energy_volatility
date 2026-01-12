from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import time

from proj.features import transforms, anomaly_detection
from proj.utils.dates import *

ET = ZoneInfo("America/New_York")

# ----------------------------
# Small utilities
# ----------------------------


def _as_daily(df: pd.DataFrame, freq: str, how: str = "last") -> pd.DataFrame:
    """
    Resample to a daily-like frequency while preserving a DatetimeIndex.
    """
    if how == "last":
        return df.resample(freq).last()
    if how == "mean":
        return df.resample(freq).mean()
    raise ValueError(f"_as_daily: unknown how='{how}'")


def _make_lags(df: pd.DataFrame, cols: Sequence[str], max_lag: int) -> pd.DataFrame:
    """
    Add lagged versions of selected columns: col_lag1 ... col_lagK
    """
    out = df.copy()
    for lag in range(1, max_lag + 1):
        for c in cols:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
    return out


def _winsorize_series(s: pd.Series, p: float) -> pd.Series:
    # Prefer transforms.winsorize if you already have it; fall back to quantile clip.
    if hasattr(transforms, "winsorize"):
        return transforms.winsorize(s, p)
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)

def drop_outlier_intraday_returns(df: pd.DataFrame, cols: list[str], max_abs_return: float) -> pd.DataFrame:
    """
    Drops rows where ANY of the specified return columns exceeds max_abs_return in absolute value.
    Returns are assumed to be in percent if you multiply by 100.
    """
    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= df[c].abs() <= max_abs_return
    return df.loc[mask]


# ----------------------------
# Equities: intraday -> daily realized measures
# ----------------------------

# --- intraday equities preprocessing (updated to ET trading-day buckets + ET-close UTC identity)



def preprocess_equities(
    df: pd.DataFrame,
    base_features_cfg: dict,
    source_cfg: dict,
) -> pd.DataFrame:
    """Daily realized measures from intraday prices; index stamped to ET close in UTC."""

    # =============================================================================
    # 1) Config
    # =============================================================================
    freq = base_features_cfg.get("resample_freq", "1D")          # daily buckets
    window = int(source_cfg.get("idio_window", 60))
    min_bins = int(source_cfg.get("min_intraday_bins", 1))
    log_eps = float(source_cfg.get("log_eps", 1e-12))
    close_et = time(*map(int, (source_cfg.get("market_close_et", "16:00")).split(":")))
    max_abs = float(source_cfg.get("max_abs_intraday_ret_pct", 5.0))  
    

    # =============================================================================
    # 2) Validate + standardize index (tz-aware UTC)
    # =============================================================================
    out = df.copy().sort_index()
    out = ensure_datetime_index_utc(out)
    out = filter_rth(out)


    # =============================================================================
    # 3) Intraday features (returns, idio residuals, RV components)
    # =============================================================================
    out["XLE_r"] = transforms.log_returns(out, "XLE") * 100
    out["SPY_r"] = transforms.log_returns(out, "SPY") * 100

    # still drop remaining extreme returns (extra safety net)
    out = drop_outlier_intraday_returns(out, ["XLE_r", "SPY_r"], max_abs)
    out = out.dropna(subset=["XLE_r", "SPY_r"])

    out = transforms.estimate_idiosyncratic(out, window=window)
    out = out.dropna(subset=["XLE_idio"])

    out["xle_r_sq"] = out["XLE_r"] ** 2
    out["spy_r_sq"] = out["SPY_r"] ** 2
    out["idio_sq"]  = out["XLE_idio"] ** 2



    # =============================================================================
    # 5) Daily aggregation by ET trading day (not UTC midnight)
    # =============================================================================
    out_et = out.tz_convert(ET)

    daily = out_et.resample(freq).agg(
        rv_xle=("xle_r_sq", "sum"),
        rv_spy=("spy_r_sq", "sum"),
        rv_idio=("idio_sq", "sum"),
        ret_xle=("XLE_r", "sum"),
        ret_spy=("SPY_r", "sum"),
        ret_idio=("XLE_idio", "sum"),
        n_intra=("XLE_r", "count"),
    )

    # =============================================================================
    # 6) Coverage filter
    # =============================================================================
    daily = daily.loc[daily["n_intra"] >= min_bins].copy()

    # =============================================================================
    # 7) Stamp daily identity to ET close converted to UTC (gold standard)
    #    - resample produces an ET-midnight label; convert that label -> ET close -> UTC
    # =============================================================================

    daily = standardize_daily_identity_index(daily, close_et=close_et)

    # =============================================================================
    # 8) Derived daily features (logs + realized vol)
    # =============================================================================
    daily["log_rv_xle"]  = np.log(daily["rv_xle"]  + log_eps)
    daily["log_rv_spy"]  = np.log(daily["rv_spy"]  + log_eps)
    daily["log_rv_idio"] = np.log(daily["rv_idio"] + log_eps)

    daily["rvol_xle"]  = np.sqrt(daily["rv_xle"].clip(lower=0))
    daily["rvol_spy"]  = np.sqrt(daily["rv_spy"].clip(lower=0))
    daily["rvol_idio"] = np.sqrt(daily["rv_idio"].clip(lower=0))

    # =============================================================================
    # 9) Final column order
    # =============================================================================
    cols = [
        "ret_xle", "ret_spy", "ret_idio",
        "rv_xle", "rv_spy", "rv_idio",
        "log_rv_xle", "log_rv_spy", "log_rv_idio",
        "rvol_xle", "rvol_spy", "rvol_idio",
        "n_intra",
    ]
    return daily[cols].dropna()



# ----------------------------
# Faily Equities & Macro:  ETF proxies + vol indices -> lagged exog
# ----------------------------

def preprocess_equities_daily(
    df: pd.DataFrame,
    base_features_cfg: dict,
    source_cfg: dict,
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("preprocess_equities_daily: df must be a non-empty DataFrame")
    ensure_datetime_index_utc(df)

    out = df.copy().sort_index()

    # ---- config
    freq = base_features_cfg.get("resample_freq", "1D")
    max_lag = int(base_features_cfg.get("max_lag", 1))
    log_eps = float(base_features_cfg.get("log_eps", 1e-12))

    # IMPORTANT: do NOT asfreq+ffill before returns
    align_business_days = bool(source_cfg.get("align_business_days", False))  # default False
    ffill_rates = bool(source_cfg.get("ffill_rates", True))
    winsor_p = float(source_cfg.get("winsor_p", 0.005))

    price_cols = list(source_cfg.get("price_cols", []))
    vol_cols = list(source_cfg.get("vol_cols", []))
    rate_cols = list(source_cfg.get("rate_cols", []))

    add_absrets = bool(base_features_cfg.get("add_absrets", True))
    add_vol_logs = bool(base_features_cfg.get("add_vol_logs", True))
    winsorize_cols = list(base_features_cfg.get("winsorize_cols", []))

    close_et = time(*map(int, (source_cfg.get("market_close_et", "16:00")).split(":")))
    # Resample to daily close, keep only observed days
    # out = _as_daily(out, freq=freq, how="last")
    out = standardize_daily_identity_index(out, close_et=close_et)

    # ---- rates: forward-fill (slow-moving, ok)
    if rate_cols:
        for c in rate_cols:
            if c in out.columns:
                out[c] = out[c].astype(float)
        if ffill_rates:
            out[rate_cols] = out[rate_cols].ffill()

    # ---- price returns (+ abs returns) on observed days ONLY
    for c in price_cols:
        if c not in out.columns:
            continue
        s = out[c].astype(float)

        # allow missing prices (don’t ffill here)
        # if non-positive values exist, drop them (bad data) rather than raising
        s = s.where(s > 0)

        out[f"{c}_ret"] = transforms.log_returns(out, c) * 100
        if add_absrets:
            out[f"{c}_absret"] = out[f"{c}_ret"].abs()

    # ---- vol index -> log level (on observed days)
    if add_vol_logs:
        for c in vol_cols:
            if c not in out.columns:
                continue
            s = out[c].astype(float).where(out[c].astype(float) > 0)
            out[f"log_{c}"] = np.log(s + log_eps)

    # winsorize heavy tails
    for c in winsorize_cols:
        if c in out.columns:
            out[c] = _winsorize_series(out[c], winsor_p)

    # collect base exog features
    exog_base: list[str] = []
    for c in rate_cols:
        if c in out.columns:
            exog_base.append(c)

    for c in price_cols:
        if f"{c}_ret" in out.columns:
            exog_base.append(f"{c}_ret")
        if add_absrets and f"{c}_absret" in out.columns:
            exog_base.append(f"{c}_absret")

    if add_vol_logs:
        for c in vol_cols:
            lc = f"log_{c}"
            if lc in out.columns:
                exog_base.append(lc)

    # dedupe
    seen = set()
    exog_base = [c for c in exog_base if not (c in seen or seen.add(c))]

    exog = out[exog_base]

    # Optional: align to business-day grid AFTER feature computation
    # (This will introduce NaNs on holidays, which is fine; do NOT ffill returns.)
    if align_business_days:
        exog = exog.asfreq("B")

        # If you truly want to ffill only rate levels on this grid:
        if ffill_rates and rate_cols:
            present_rates = [c for c in rate_cols if c in exog.columns]
            exog[present_rates] = exog[present_rates].ffill()


    # CRITICAL: do NOT dropna here
    return exog





# ----------------------------
# Weather: daily -> anomaly features -> lag (drop contemporaneous)
# ----------------------------

def preprocess_weather(
    df: pd.DataFrame,
    base_features_cfg: dict,
    weather_cfg: dict,
) -> pd.DataFrame:
    """
    Weather preprocessing for anomaly-based exogenous features:
      - daily alignment (asfreq)
      - per-feature z-score anomalies (+ abs)
      - optional multivariate IsolationForest anomaly score/flag
      - lag anomalies and drop contemporaneous
    """
    ensure_datetime_index_utc(df)
    out = df.copy().sort_index()

    # ---- config
    freq = base_features_cfg.get("resample_freq", "1D")
    max_lag = int(base_features_cfg.get("max_lag", 1))
    eps = float(base_features_cfg.get("eps", 1e-8))

    z_method = weather_cfg.get("zscore_method", "doy_robust")
    rolling_window = int(weather_cfg.get("rolling_window", 30))

    enable_iforest = bool(weather_cfg.get("enable_iforest", True))
    contamination = float(weather_cfg.get("contamination", 0.01))
    random_state = int(base_features_cfg.get("random_state", 99))
    
    close_et = time(*map(int, (weather_cfg.get("market_close_et", "16:00")).split(":")))

    # 1) align to daily frequency (weather is naturally daily)
    out = out.asfreq(freq)

    cols = list(out.columns)
    if not cols:
        raise ValueError("preprocess_weather: df has no weather columns")

    # 2) z anomalies per feature
    for c in cols:
        z = anomaly_detection.compute_zscore_anomaly(
            out[c],
            method=z_method,
            rolling_window=rolling_window,
            eps=eps,
        )
        out[f"{c}_anom_z"] = z
        out[f"{c}_anom_absz"] = z.abs()

    # 3) multivariate isolation forest on z-space
    if enable_iforest:
        z_cols = [f"{c}_anom_z" for c in cols]

        iso_out = anomaly_detection.compute_isolated_forest_anomaly(
            df=out,
            cols=z_cols,
            contamination=contamination,   # define in your config or set a default
            random_state=random_state,     # define in your config or set a default
        )

        # attach the multivariate outputs (single set of columns)
        out["weather_iforest_score"] = iso_out["weather_iforest_score"]
        out["weather_iforest_flag"]  = iso_out["weather_iforest_flag"]

        
    # 4) lag anomaly features only, then drop contemporaneous
    # base_weather_cols = [c for c in cols if c in out.columns]

    # anom_cols = (
    #     [f"{c}_anom_z" for c in cols] +
    #     [f"{c}_anom_absz" for c in cols]
    # )

    # lag_cols = [c for c in (base_weather_cols + anom_cols + iforest_cols) if c in out.columns]

    # out = _make_lags(out, lag_cols, max_lag=max_lag)
    # out = out.drop(columns=lag_cols)
    out = standardize_daily_identity_index(out, close_et=close_et)
    return out.dropna()


# ----------------------------
# Convenience helpers
# ----------------------------

def filter_rth(df):
    et = df.tz_convert("America/New_York")
    mask = (
        (et.index.dayofweek < 5) &
        (et.index.time >= pd.Timestamp("09:30").time()) &
        (et.index.time <= pd.Timestamp("16:00").time())
    )
    return df.loc[mask]

def long_to_wide(df: pd.DataFrame, pivot_by: str = "close") -> pd.DataFrame:
    """
    Convert long format with columns ['date','Symbol',<pivot_by>] to wide format.
    """
    wide = df.pivot(index="date", columns="Symbol", values=pivot_by).reset_index()
    wide.columns.name = None
    return wide


