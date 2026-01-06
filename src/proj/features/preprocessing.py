from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from proj.features import transforms, anomaly_detection


# ----------------------------
# Small utilities
# ----------------------------

def _require_datetime_index(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: df must be indexed by a DatetimeIndex")


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


# ----------------------------
# Equities: intraday -> daily realized measures
# ----------------------------

def preprocess_equities(
    df: pd.DataFrame,
    base_features_cfg: dict,
    source_cfg: dict,
) -> pd.DataFrame:
    """
    Build DAILY realized measures from 30-min prices for XLE/SPY.

    Input:
      - df: 30-min price dataframe, DatetimeIndex, columns include ["XLE","SPY"].

    Output (daily index):
      - ret, mkt_ret
      - rv_total, rv_idio
      - log_rv_total, log_rv_idio
      - rvol_total, rvol_idio
      - n_intra
    """
    _require_datetime_index(df, "preprocess_equities")
    out = df.copy().sort_index()

    # ---- config
    freq = base_features_cfg.get("resample_freq", "1D")
    window = int(source_cfg.get("idio_window", 60))
    min_bins = int(source_cfg.get("min_intraday_bins", 1))
    log_eps = float(source_cfg.get("log_eps", 1e-12))

    # 1) Intraday log returns
    out["XLE_r"] = transforms.log_returns(out, "XLE") * 100
    out["SPY_r"] = transforms.log_returns(out, "SPY") * 100
    out = out.dropna(subset=["XLE_r", "SPY_r"])

    # 2) Intraday idiosyncratic residuals
    out = transforms.estimate_idiosyncratic(out, window=window)
    out = out.dropna(subset=["XLE_idio"])

    # 3) Intraday components for realized measures
    out["xle_r_sq"] = out["XLE_r"] ** 2
    out["spy_r_sq"] = out["SPY_r"] ** 2
    out["idio_sq"]  = out["XLE_idio"] ** 2

    # 4) Daily aggregation
    daily = out.resample(freq).agg(
        rv_xle=("xle_r_sq", "sum"),
        rv_spy=("spy_r_sq", "sum"),
        rv_idio=("idio_sq", "sum"),
        ret_xle=("XLE_r", "sum"),
        ret_spy=("SPY_r", "sum"),
        ret_idio=("XLE_idio", "sum"),
        n_intra=("XLE_r", "count"),
    )

    # 5) Filter low-coverage days
    daily = daily.loc[daily["n_intra"] >= min_bins].copy()

    # 6) Logs + realized vol (sqrt RV)
    daily["log_rv_xle"]  = np.log(daily["rv_xle"]  + log_eps)
    daily["log_rv_spy"]  = np.log(daily["rv_spy"]  + log_eps)
    daily["log_rv_idio"] = np.log(daily["rv_idio"] + log_eps)

    daily["rvol_xle"]  = np.sqrt(daily["rv_xle"].clip(lower=0))
    daily["rvol_spy"]  = np.sqrt(daily["rv_spy"].clip(lower=0))
    daily["rvol_idio"] = np.sqrt(daily["rv_idio"].clip(lower=0))

    # 7) Tidy order
    cols = [
        "ret_xle", "ret_spy", "ret_idio",
        "rv_xle", "rv_spy", "rv_idio",
        "log_rv_xle", "log_rv_spy", "log_rv_idio",
        "rvol_xle", "rvol_spy", "rvol_idio",
        "n_intra",
    ]
    return daily[cols].dropna()



# ----------------------------
# Equities substitutes: daily ETF proxies + vol indices -> lagged exog
# ----------------------------

def preprocess_equities_daily(
    df: pd.DataFrame,
    base_features_cfg: dict,
    source_cfg: dict,
) -> pd.DataFrame:
    """
    Build lagged DAILY exogenous features from ETF proxies (prices) and vol indices (levels).

    Expected input:
      - df: daily wide df (DatetimeIndex) with levels for:
          price_cols (e.g., HYG, TLT, UNG, USO) and vol_cols (e.g., ^VIX, ^OVX).

    Output:
      - lagged features only (drops contemporaneous) unless keep_raw_levels=True
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("preprocess_equities_daily: df must be a non-empty DataFrame")
    _require_datetime_index(df, "preprocess_equities_daily")

    out = df.copy().sort_index()

    # ---- config
    freq = base_features_cfg.get("resample_freq", "1D")
    max_lag = int(base_features_cfg.get("max_lag", 1))
    log_eps = float(base_features_cfg.get("log_eps", 1e-12))

    use_business_days = bool(source_cfg.get("use_business_days", True))
    ffill = bool(source_cfg.get("ffill", True))
    winsor_p = float(source_cfg.get("winsor_p", 0.005))

    price_cols = list(source_cfg.get("price_cols", ["HYG", "TLT", "UNG", "USO"]))
    vol_cols = list(source_cfg.get("vol_cols", ["^VIX", "^OVX"]))

    add_absrets = bool(base_features_cfg.get("add_absrets", True))
    add_vol_logs = bool(base_features_cfg.get("add_vol_logs", True))
    winsorize_cols = list(base_features_cfg.get("winsorize_cols", ["UNG_ret", "USO_ret"]))
    keep_raw_levels = bool(base_features_cfg.get("keep_raw_levels", False))

    # 1) normalize daily calendar
    out = _as_daily(out, freq=freq, how="last")
    if use_business_days:
        out = out.asfreq("B")
    if ffill:
        out = out.ffill()

    # 2) price -> log returns (+ abs returns)
    for c in price_cols:
        if c not in out.columns:
            continue
        if (out[c] <= 0).any():
            raise ValueError(f"preprocess_equities_daily: '{c}' has non-positive values; cannot log.")
        out[f"{c}_ret"] = np.log(out[c]).diff()
        if add_absrets:
            out[f"{c}_absret"] = out[f"{c}_ret"].abs()

    # 3) vol index -> log level
    if add_vol_logs:
        for c in vol_cols:
            if c not in out.columns:
                continue
            if (out[c] <= 0).any():
                raise ValueError(f"preprocess_equities_daily: '{c}' has non-positive values; cannot log.")
            out[f"log_{c}"] = np.log(out[c] + log_eps)

    # 4) winsorize heavy-tailed daily returns (common for UNG/USO)
    for c in winsorize_cols:
        if c in out.columns:
            out[c] = _winsorize_series(out[c], winsor_p)

    # 5) select contemporaneous exog base features to lag
    exog_base: list[str] = []
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

    # dedupe, preserve order
    seen = set()
    exog_base = [c for c in exog_base if not (c in seen or seen.add(c))]

    # 6) create lags + drop contemporaneous by default
    # out = _make_lags(out, exog_base, max_lag=max_lag)
    # if exog_base:
    #     out = out.drop(columns=exog_base)

    # 7) optional: keep raw levels (debug/EDA only)
    # if not keep_raw_levels:
    #     drop_raw = [c for c in price_cols + vol_cols if c in out.columns]
    #     if drop_raw:
    #         out = out.drop(columns=drop_raw)

    return out[exog_base].dropna()


# ----------------------------
# Macro: resample -> ffill -> lag (drop contemporaneous)
# ----------------------------

def preprocess_macro(
    df: pd.DataFrame,
    base_features_cfg: dict,
    source_cfg: dict,
) -> pd.DataFrame:
    """
    Simple macro preprocessing:
      - resample to freq
      - forward-fill
      - create lags
      - drop contemporaneous columns
    """
    _require_datetime_index(df, "preprocess_macro")
    out = df.copy().sort_index()

    # ---- config
    freq = base_features_cfg.get("resample_freq", "1D")
    max_lag = int(base_features_cfg.get("max_lag", 1))
    ffill = bool(source_cfg.get("ffill", True))

    # 1) resample
    out = _as_daily(out, freq=freq, how="last")

    # 2) ffill
    if ffill:
        out = out.ffill()

    # 3) lag + drop contemporaneous
    # exog_base = list(out.columns)
    # out = _make_lags(out, exog_base, max_lag=max_lag)
    # out = out.drop(columns=exog_base)

    return out.dropna()


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
    _require_datetime_index(df, "preprocess_weather")
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

    return out.dropna()


# ----------------------------
# Convenience helpers
# ----------------------------

def long_to_wide(df: pd.DataFrame, pivot_by: str = "close") -> pd.DataFrame:
    """
    Convert long format with columns ['date','Symbol',<pivot_by>] to wide format.
    """
    wide = df.pivot(index="date", columns="Symbol", values=pivot_by).reset_index()
    wide.columns.name = None
    return wide


def preprocess_for_vol_prediction(
    df: pd.DataFrame,
    exog_cols: Sequence[str],
    target_cols: Sequence[str],
    lag: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare exogenous regressors for volatility prediction by:
      - enforcing stationarity (via transforms.enforce_stationarity)
      - lagging to avoid look-ahead bias
      - concatenating with targets and dropping NaNs at the end

    Note: scaling is intentionally omitted here; many volatility models
    (and your current pipeline) do fine without it. Add scaling outside
    if you explicitly need it.
    """
    out = df.copy()

    X_raw = out[list(exog_cols)]
    X_stat = transforms.enforce_stationarity(X_raw)
    X_lagged = X_stat.shift(lag)

    combined = pd.concat([out[list(target_cols)], X_lagged], axis=1).dropna()

    X = combined[X_lagged.columns]
    y = combined[list(target_cols)]
    return X, y
