

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

WEATHER_FEATURES = [
    "temperature_2m_mean",
    "apparent_temperature_mean",
    "precipitation_sum",
    "precipitation_hours",
    "snowfall_sum",
    "wind_speed_10m_mean",
    "wind_gusts_10m_mean",
    "cloud_cover_mean",
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
]

def compute_zscore_anomaly(
    s: pd.Series,
    method: str = "doy_robust",   # "rolling" also supported below
    rolling_window: int = 30,
    eps: float = 1e-8,
) -> pd.Series:
    """
    Return a z-score anomaly series.
    - doy_robust: removes seasonality via day-of-year median and scales via MAD (robust)
    - rolling: rolling z-score (weaker for seasonal series)
    """
    s = s.copy()

    if method == "rolling":
        mu = s.rolling(rolling_window, min_periods=max(5, rolling_window // 3)).mean()
        sd = s.rolling(rolling_window, min_periods=max(5, rolling_window // 3)).std()
        return (s - mu) / (sd + eps)

    if method == "doy_robust":
        doy = s.index.dayofyear

        # seasonal baseline: median per day-of-year
        med = s.groupby(doy).transform("median")

        # robust scale: MAD per day-of-year
        def _mad(x: pd.Series) -> float:
            m = np.nanmedian(x)
            return np.nanmedian(np.abs(x - m))

        mad = s.groupby(doy).transform(_mad)

        # 1.4826 * MAD ~ std under normality
        scale = 1.4826 * mad

        resid = s - med
        return resid / (scale + eps)

    raise ValueError(f"Unknown method: {method}")


def compute_isolated_forest_anomaly(
    df: pd.DataFrame,
    cols: list[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit IsolationForest on the provided columns and return:
      - anomaly_score (higher = more anomalous)
      - anomaly_flag (1 = anomaly)
    Assumes df is daily and already de-seasonalized (recommended).
    """
    X = df[cols].astype(float)

    # simple missing handling: time interpolation, then back/forward fill edges
    X = X.interpolate(limit=3).ffill().bfill()

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    # sklearn: score_samples -> higher is "more normal"
    normal_score = model.score_samples(X)
    anomaly_score = -normal_score  # flip so higher = more anomalous
    anomaly_flag = (model.predict(X) == -1).astype(int)

    out = df.copy()
    out["weather_iforest_score"] = anomaly_score
    out["weather_iforest_flag"] = anomaly_flag
    return out


