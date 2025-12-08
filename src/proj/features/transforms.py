import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def difference(series, d=1):
    s = series
    for _ in range(d):
        s = s.diff().dropna()
    return s


def log_returns(df, column):
    """Compute clean log returns for a price column."""
    
    # compute raw log returns
    r = np.log(df[column] / df[column].shift(1))
    
    # remove problematic values
    r = r.replace([np.inf, -np.inf], np.nan)   # log(0) or div-by-zero cases
    
    # optional: drop NaNs (usually first row)
    # r = r.dropna()
    
    return r


def realized_vol(time_series, window: int = 21, annualize: bool = False) -> pd.DataFrame:
    """Compute rolling standard deviation of daily returns (realized volatility)."""
    
    time_series = time_series.copy()
    volatility = time_series.rolling(window).std()
    
    if annualize:
        volatility *= np.sqrt(256)
    
    return volatility 


def realized_var(time_series, window: int = 21, annualize: bool = False) -> pd.DataFrame:
    """Compute rolling standard deviation of daily returns (realized volatility)."""
    
    time_series = time_series.copy()
    volatility = time_series.rolling(window).var()
    
    if annualize:
        volatility *= np.sqrt(256)
    
    return volatility




def enforce_stationarity(X, max_differencing = 4, threshold = .05):

    '''
    function that makes a data frame of exogoenous features stationary by taking the difference. 
    '''
    stationary_df = pd.DataFrame()

    stationary_df = pd.DataFrame(index=X.index)

    for col in X.columns:
        time_series = X[col].copy()

        # Drop NaNs for checks
        ts_nonan = time_series.dropna()

        # 1) If too few observations or constant -> skip this column
        if  ts_nonan.nunique() <= 1:
            # Option A: drop column silently
            # print(f"Skipping column {col}: constant or too few observations.")
            continue

        d = 0

        while d < max_differencing:
            ts_nonan = time_series.dropna()

            try:
                adf, p_val, lag, nobs, cv, _ = adfuller(ts_nonan)
            except ValueError:
                # Catch any weird cases (e.g., still constant)
                break

            if p_val < threshold:
                # Stationary -> stop differencing
                break

            # Otherwise, difference and try again
            time_series = time_series.diff()
            d += 1

        col_name = f"{col}_d{d}"
        stationary_df[col_name] = time_series  # keep NaNs; drop later in pipeline

    return stationary_df