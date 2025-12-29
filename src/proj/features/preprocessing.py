import pandas as pd
from sklearn.preprocessing import StandardScaler
from proj.features import transforms, anomaly_detection
import numpy as np 


def preprocess_equities(df: pd.DataFrame, base_features_cfg: dict,  source_cfg: dict) -> pd.DataFrame:
    """
    Input:
      df: 30-min price dataframe with a DatetimeIndex (timezone-aware recommended),
          containing columns ['XLE', 'SPY'] as prices.

    Output:
      daily dataframe indexed by date with columns suitable for:
        - standard GARCH on daily returns (ret)
        - realized-vol models / realized-GARCH (rv_* columns)
    """
    df = df.copy()

    # =========== config defaults ===========
    window = int(source_cfg.get("idio_window", 60))
    freq = base_features_cfg['resample_freq']
    min_bins = int(source_cfg.get("min_intraday_bins", 1))  # e.g., require at least 8 half-hours
     # =======================================




    # 1) Intraday log returns
    df["XLE_r"] = transforms.log_returns(df, "XLE")
    df["SPY_r"] = transforms.log_returns(df, "SPY")

    df = df.dropna(subset=["XLE_r", "SPY_r"])

    # 2) Idiosyncratic residuals (rolling CAPM on intraday returns)
    df = transforms.estimate_idiosyncratic(df, window=window)  # must create df["XLE_idio"]

    df = df.dropna(subset=["XLE_idio"])

    # 3) Intraday components for realized measures
    df["xle_r_sq"] = df["XLE_r"] ** 2
    df["idio_sq"] = df["XLE_idio"] ** 2

    # 4)  aggregation
    daily = df.resample(freq).agg(
        rv_total=("xle_r_sq", "sum"),
        rv_idio=("idio_sq", "sum"),
        mkt_ret=("SPY_r", "sum"),     # daily SPY close-to-close log return from intraday
        ret=("XLE_r", "sum"),         # daily XLE close-to-close log return from intraday
        n_intra=("XLE_r", "count"),
    )

    # 5) Basic day-quality filtering (optional but recommended)
    daily = daily[daily["n_intra"] >= min_bins].copy()

    # 6) Logs (do AFTER daily aggregation)
    eps = float(source_cfg.get("log_eps", 1e-12))
    daily["log_rv_total"] = np.log(daily["rv_total"] + eps)
    daily["log_rv_idio"] = np.log(daily["rv_idio"] + eps)

    # Optional: realized vol (sqrt RV) if you prefer in volatility units
    daily["rvol_total"] = np.sqrt(daily["rv_total"].clip(lower=0))
    daily["rvol_idio"] = np.sqrt(daily["rv_idio"].clip(lower=0))

    # Keep a tidy column order
    cols = [
        "ret", "mkt_ret",
        "rv_total", "rv_idio",
        "log_rv_total", "log_rv_idio",
        "rvol_total", "rvol_idio",
        "n_intra",
    ]
    daily = daily[cols].dropna()

    return daily


def preprocess_equities_daily(
    df: pd.DataFrame,
    base_features_cfg: dict,
    source_cfg: dict,
) -> pd.DataFrame:
    """
    Prepare DAILY features for volatility modeling (GARCH-X / HAR-RV-X / realized-vol models).

    Expected input:
      - df: daily wide dataframe indexed by timestamp (DatetimeIndex), columns are tickers, e.g.
            ['HYG','TLT','UNG','USO','^VIX','^OVX'] with *levels* (prices / index levels).
      - base_features_cfg: config controlling which derived features to create (optional).
      - source_cfg: config for lags, winsorization, business day alignment, etc.

    Output:
      - daily dataframe with derived columns (returns for ETFs, log-levels for vol indices),
        and lagged versions suitable for exogenous regressors.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("preprocess_equities_daily: df must be a non-empty DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("preprocess_equities_daily: df must be indexed by a DatetimeIndex")

    out = df.copy().sort_index()

    # =========== config defaults ===========
    use_business_days = bool(source_cfg.get("use_business_days", True))
    ffill = bool(source_cfg.get("ffill", True))
    winsor_p = float(source_cfg.get("winsor_p", 0.005))

    max_lag = int(base_features_cfg.get("max_lag", 1))
    log_eps = float(base_features_cfg.get("log_eps", 1e-12))

    # =======================================



    # Which columns are "prices" (transform to log returns)
    price_cols = source_cfg.get("price_cols", ["HYG", "TLT", "UNG", "USO"])
    # Which columns are vol indices (transform to log levels)
    vol_cols = source_cfg.get("vol_cols", ["^VIX", "^OVX"])

    # Optional feature toggles
    add_absrets = bool(base_features_cfg.get("add_absrets", True))
    add_levels_logs = bool(base_features_cfg.get("add_vol_logs", True))
    winsorize_cols = base_features_cfg.get("winsorize_cols", ["UNG_ret", "USO_ret"])

    # 1) align to business days (typical for US ETFs)
    if use_business_days:
        out = out.asfreq("B")
    if ffill:
        out = out.ffill()

    # 2) build log returns for ETF proxies (stationary)
    for c in price_cols:
        if c not in out.columns:
            continue
        # guard against non-positive values before log
        if (out[c] <= 0).any():
            raise ValueError(f"preprocess_equities_daily: column '{c}' has non-positive values; cannot log.")
        out[f"{c}_ret"] = np.log(out[c]).diff()
        if add_absrets:
            out[f"{c}_absret"] = out[f"{c}_ret"].abs()

    # 3) log-levels for vol indices (persistent regimes)
    if add_levels_logs:
        for c in vol_cols:
            if c not in out.columns:
                continue
            if (out[c] <= 0).any():
                raise ValueError(f"preprocess_equities_daily: column '{c}' has non-positive values; cannot log.")
            out[f"log_{c}"] = np.log(out[c] + log_eps)


    for c in winsorize_cols:
        if c in out.columns:
            out[c] = transforms.winsorize(out[c], winsor_p)

    # 5) create lagged exogenous versions (avoid look-ahead bias)
    # Choose exog base cols from what exists
    exog_base = []

    for c in price_cols:
        if f"{c}_ret" in out.columns:
            exog_base.append(f"{c}_ret")
        if add_absrets and f"{c}_absret" in out.columns:
            exog_base.append(f"{c}_absret")

    if add_levels_logs:
        for c in vol_cols:
            lc = f"log_{c}"
            if lc in out.columns:
                exog_base.append(lc)

    # Deduplicate while preserving order
    seen = set()
    exog_base = [c for c in exog_base if not (c in seen or seen.add(c))]

    for lag in range(1, max_lag + 1):
        for c in exog_base:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)

    # 6) keep only what you want to feed into models (plus optional diagnostics)
    keep_raw = bool(base_features_cfg.get("keep_raw_levels", False))
    keep_cols = []

    if keep_raw:
        keep_cols += [c for c in price_cols + vol_cols if c in out.columns]

    keep_cols += [c for c in out.columns if c.endswith("_ret") or c.endswith("_absret") or c.startswith("log_")]
    keep_cols += [c for c in out.columns if "_lag" in c]

    keep_cols = sorted(set(keep_cols), key=lambda x: list(out.columns).index(x))

    out = out[keep_cols].dropna()

    return out

def preprocess_macro(df: pd.DataFrame, base_features_cfg: dict,  source_cfg: dict):

    df = df.copy().sort_index()

    # =========== config defaults ===========
    max_lag = int(base_features_cfg.get("max_lag", 1))
    freq = base_features_cfg.get("resample_freq", "1D")
    ffill = bool(source_cfg.get("ffill", True))
    # =======================================

    # 1) Resample to target frequency
    df = df.resample(freq).last()

    # 2) Forward-fill (represents information availability)
    if ffill:
        df = df.ffill()

    # 3) Create lagged features (avoid look-ahead bias)
    for lag in range(1, max_lag + 1):
        for col in df.columns:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # 4) Drop rows with insufficient history
    df = df.dropna()

    return df

    return df.ffill()



def preprocess_weather(
    df: pd.DataFrame,
    base_features_cfg: dict, 
    weather_cfg: dict,
) -> pd.DataFrame:
    """
    End-to-end daily preprocessing:
      - aligns daily
      - builds z-score anomalies per feature (seasonal robust by default)
      - builds multivariate isolation forest anomaly score/flag
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("weather df must have a DatetimeIndex")

    df = df.copy().sort_index()

    # =========== config defaults ===========
    max_lag = int(base_features_cfg.get("max_lag", 1))
    freq = base_features_cfg.get("resample_freq", "1D")

    eps = float(base_features_cfg.get("eps", 1e-8))
    z_method = weather_cfg.get("zscore_method", "doy_robust")
    roll = int(weather_cfg.get("rolling_window", 30))

    enable_iforest = weather_cfg.get("enable_iforest", True)
    contamination = weather_cfg.get("contamination", .01)
    random_state = base_features_cfg.get("random_state", 99)


    # =======================================
    # align daily
    df = df.asfreq(freq)

    # select columns present
    cols = df.columns
    if len(cols) == 0:
        raise ValueError("No expected weather feature columns found in df.")

    # z-score anomalies (per-feature)
    out = df.copy()
    for c in cols:
        out[f"{c}_anom_z"] = anomaly_detection.compute_zscore_anomaly(out[c], method=z_method, rolling_window=roll, eps=eps)
        out[f"{c}_anom_absz"] = out[f"{c}_anom_z"].abs()

    # isolation forest on residual-like inputs:
    # use the z-anomalies (already de-seasonalized + scaled)
    if enable_iforest:
        if_cols = [f"{c}_anom_z" for c in cols]
        out = anomaly_detection.compute_isolated_forest_anomaly(
            out,
            cols=if_cols,
            contamination=contamination,
            random_state=random_state,
        )

    # lag anomalies for exogenous use
    max_lag = int(weather_cfg.get("max_lag", 1))
    lag_cols = [f"{c}_anom_absz" for c in cols] + [f"{c}_anom_z" for c in cols]
    lag_cols = [c for c in lag_cols if c in out.columns]

    for lag in range(1, max_lag + 1):
        for c in lag_cols:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)

    return out.dropna()





def clean_macro_series(df):
    return df.reset_index(drop=True)


























def long_to_wide(df, pivot_by='close'):

    wide = df.pivot(index='date', columns='Symbol', values=pivot_by)
    wide = wide.reset_index()  
    wide.columns.name = None

    return wide

def preprocess_for_vol_prediction(df, exog_cols, target_cols, lag=1):
    """
    Prepare exogenous regressors for GARCH-X by:
    1. Enforcing stationarity (differencing)
    2. Lagging features to avoid look-ahead
    3. Scaling features
    4. Dropping NaNs AFTER all transformations
    """

    df = df.copy()

    # 1) Extract exogenous features
    X_raw = df[exog_cols]
    X_stationary = transforms.enforce_stationarity(X_raw)
    X_lagged = X_stationary.shift(lag)


    combined = pd.concat([df[target_cols], X_lagged], axis=1).dropna()

    # 5) Scale only the exogenous columns (the stationarity-enforced, lagged ones)
    exog_processed_cols = X_lagged.columns  
    # scaler = StandardScaler()
    
    # X_scaled = scaler.fit_transform(combined[exog_processed_cols])
    # X = pd.DataFrame(X_scaled,
    #                  index=combined.index,
    #                  columns=exog_processed_cols)

    X = combined[exog_processed_cols]
    
    y = combined[target_cols]

    return X, y