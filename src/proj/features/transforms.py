import numpy as np
import pandas as pd

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


def realized_vol(time_series, window: int = 21, annualize: bool = True) -> pd.DataFrame:
    """Compute rolling standard deviation of daily returns (realized volatility)."""
    
    time_series = time_series.copy()
    volatility = time_series.rolling(window).std()
    
    if annualize:
        volatility *= np.sqrt(256)
    
    return volatility 