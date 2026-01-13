from __future__ import annotations

from datetime import time
from typing import Sequence

import numpy as np
import pandas as pd

from proj.features import transforms, anomaly_detection
from proj.utils.dates import ensure_datetime_index_utc, filter_rth, standardize_daily_identity_index, parse_hhmm


# ----------------------------
# Small helpers
# ----------------------------

def log1p_eps(x: pd.Series, eps: float) -> pd.Series:
    """log(x + eps) allowing x>=0 (used for realized variance)."""
    return np.log(x.clip(lower=0) + eps)


def log_pos_eps(x: pd.Series, eps: float) -> pd.Series:
    """log(x + eps) but treat non-positive values as missing (used for indices/levels that must be >0)."""
    xx = x.astype(float)
    return np.log(xx.where(xx > 0) + eps)


def winsorize_series(s: pd.Series, p: float) -> pd.Series:
    if hasattr(transforms, "winsorize"):
        return transforms.winsorize(s, p)
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def drop_outlier_intraday_returns(df: pd.DataFrame, cols: Sequence[str], max_abs: float) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= df[c].abs() <= max_abs
    return df.loc[mask]


# ----------------------------
# Intraday equities -> daily RV
# ----------------------------

def preprocess_equities(
    df: pd.DataFrame,
    base_features_cfg: dict,
    source_cfg: dict,
) -> pd.DataFrame:
    freq = base_features_cfg.get("resample_freq", "1D")
    window = int(source_cfg.get("idio_window", 60))
    min_bins = int(source_cfg.get("min_intraday_bins", 1))
    eps = float(source_cfg.get("log_eps", 1e-12))
    close_et = parse_hhmm(source_cfg.get("market_close_et", "16:00"))
    max_abs = float(source_cfg.get("max_abs_intraday_ret_pct", 5.0))

    out = ensure_datetime_index_utc(df, name="equities_intraday").copy().sort_index()
    out = filter_rth(out, name="equities_intraday")

    # returns (pct)
    out["XLE_r"] = transforms.log_returns(out, "XLE") * 100
    out["SPY_r"] = transforms.log_returns(out, "SPY") * 100
    out = drop_outlier_intraday_returns(out, ["XLE_r", "SPY_r"], max_abs).dropna(subset=["XLE_r", "SPY_r"])

    # idio residual return + squares
    out = transforms.estimate_idiosyncratic(out, window=window).dropna(subset=["XLE_idio"])
    out["xle_r_sq"] = out["XLE_r"] ** 2
    out["spy_r_sq"] = out["SPY_r"] ** 2
    out["idio_sq"]  = out["XLE_idio"] ** 2

    # aggregate by local trading day (ET) then stamp identity to ET close in UTC
    daily = (
        out.tz_convert("America/New_York")
           .resample(freq)
           .agg(
                rv_xle=("xle_r_sq", "sum"),
                rv_spy=("spy_r_sq", "sum"),
                rv_idio=("idio_sq", "sum"),
                ret_xle=("XLE_r", "sum"),
                ret_spy=("SPY_r", "sum"),
                ret_idio=("XLE_idio", "sum"),
                n_intra=("XLE_r", "count"),
            )
    )

    daily = daily.loc[daily["n_intra"] >= min_bins].copy()
    daily = standardize_daily_identity_index(daily, close_et=close_et)

    # log + rvol 
    daily["log_rv_xle"]  = log1p_eps(daily["rv_xle"],  eps)
    daily["log_rv_spy"]  = log1p_eps(daily["rv_spy"],  eps)
    daily["log_rv_idio"] = log1p_eps(daily["rv_idio"], eps)

    daily["rvol_xle"]  = np.sqrt(daily["rv_xle"].clip(lower=0))
    daily["rvol_spy"]  = np.sqrt(daily["rv_spy"].clip(lower=0))
    daily["rvol_idio"] = np.sqrt(daily["rv_idio"].clip(lower=0))

    cols = [
        "ret_xle", "ret_spy", "ret_idio",
        "rv_xle", "rv_spy", "rv_idio",
        "log_rv_xle", "log_rv_spy", "log_rv_idio",
        "rvol_xle", "rvol_spy", "rvol_idio",
        "n_intra",
    ]
    return daily[cols].dropna()


# ----------------------------
# Daily ETFs/macro -> exog features
# ----------------------------

def preprocess_equities_daily(
    df: pd.DataFrame,
    base_features_cfg: dict,
    source_cfg: dict,
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("preprocess_equities_daily: df must be a non-empty DataFrame")

    freq = base_features_cfg.get("resample_freq", "1D")
    eps = float(base_features_cfg.get("log_eps", 1e-12))

    align_business_days = bool(source_cfg.get("align_business_days", False))
    ffill_rates = bool(source_cfg.get("ffill_rates", True))
    winsor_p = float(source_cfg.get("winsor_p", 0.005))

    price_cols = list(source_cfg.get("price_cols", []))
    vol_cols   = list(source_cfg.get("vol_cols", []))
    rate_cols  = list(source_cfg.get("rate_cols", []))

    add_absrets    = bool(base_features_cfg.get("add_absrets", True))
    add_vol_logs   = bool(base_features_cfg.get("add_vol_logs", True))
    winsorize_cols = list(base_features_cfg.get("winsorize_cols", []))

    close_et = parse_hhmm(source_cfg.get("market_close_et", "16:00"))

    out = ensure_datetime_index_utc(df, name="equities_daily_raw").copy().sort_index()

    # ensure one row per day, then canonical identity
    # out = out.resample(freq).last()
    out = standardize_daily_identity_index(out, close_et=close_et)

    # rates (levels) can be ffilled
    present_rates = [c for c in rate_cols if c in out.columns]
    if present_rates:
        out[present_rates] = out[present_rates].astype(float)
        if ffill_rates:
            out[present_rates] = out[present_rates].ffill()

    # price returns (pct) on observed days (gaps in index are fine; no ffill)
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

    # vol index logs (levels must be >0)
    if add_vol_logs:
        for c in vol_cols:
            if c in out.columns:
                out[f"log_{c}"] = log_pos_eps(out[c], eps)

    # winsorize heavy tails
    for c in winsorize_cols:
        if c in out.columns:
            out[c] = winsorize_series(out[c], winsor_p)

    # build exog list (dedup while preserving order)
    exog_cols: list[str] = []
    exog_cols += [c for c in present_rates if c in out.columns]

    for c in price_cols:
        r = f"{c}_ret"
        a = f"{c}_absret"
        if r in out.columns:
            exog_cols.append(r)
        if add_absrets and a in out.columns:
            exog_cols.append(a)

    if add_vol_logs:
        for c in vol_cols:
            lc = f"log_{c}"
            if lc in out.columns:
                exog_cols.append(lc)

    seen = set()
    exog_cols = [c for c in exog_cols if not (c in seen or seen.add(c))]

    exog = out[exog_cols].copy()

    if align_business_days:
        exog = exog.asfreq("B")
        if ffill_rates and present_rates:
            rr = [c for c in present_rates if c in exog.columns]
            exog[rr] = exog[rr].ffill()

    return exog


# ----------------------------
# Weather daily -> anomaly features
# ----------------------------

def preprocess_weather(
    df: pd.DataFrame,
    base_features_cfg: dict,
    weather_cfg: dict,
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("preprocess_weather: df must be a non-empty DataFrame")

    freq = base_features_cfg.get("resample_freq", "1D")
    eps = float(base_features_cfg.get("eps", 1e-8))

    z_method = weather_cfg.get("zscore_method", "doy_robust")
    rolling_window = int(weather_cfg.get("rolling_window", 30))

    enable_iforest = bool(weather_cfg.get("enable_iforest", True))
    contamination = float(weather_cfg.get("contamination", 0.01))
    random_state = int(base_features_cfg.get("random_state", 99))

    close_et = parse_hhmm(weather_cfg.get("market_close_et", "16:00"))

    out = ensure_datetime_index_utc(df, name="weather_raw").copy().sort_index()
    out = out.asfreq(freq)

    cols = list(out.columns)
    if not cols:
        raise ValueError("preprocess_weather: df has no columns")

    # per-feature anomalies
    for c in cols:
        z = anomaly_detection.compute_zscore_anomaly(
            out[c],
            method=z_method,
            rolling_window=rolling_window,
            eps=eps,
        )
        out[f"{c}_anom_z"] = z
        out[f"{c}_anom_absz"] = z.abs()

    # multivariate iforest in z-space
    if enable_iforest:
        z_cols = [f"{c}_anom_z" for c in cols]
        iso = anomaly_detection.compute_isolated_forest_anomaly(
            df=out,
            cols=z_cols,
            contamination=contamination,
            random_state=random_state,
        )
        out["weather_iforest_score"] = iso["weather_iforest_score"]
        out["weather_iforest_flag"] = iso["weather_iforest_flag"]

    out = standardize_daily_identity_index(out, close_et=close_et)
    return out.dropna()
