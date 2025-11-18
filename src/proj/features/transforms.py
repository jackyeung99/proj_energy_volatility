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


def realized_vol(time_series, window: int = 21, annualize: bool = True) -> pd.DataFrame:
    """Compute rolling standard deviation of daily returns (realized volatility)."""
    
    time_series = time_series.copy()
    volatility = time_series.rolling(window).std()
    
    if annualize:
        volatility *= np.sqrt(256)
    
    return volatility 



def enforce_stationarity(X, max_differencing = 4, threshold = .05):

    '''
    function that makes a data frame of exogoenous features stationary by taking the difference. 
    '''
    stationary_df = pd.DataFrame()

    for col in X.columns:
        time_series = X[col].copy()
        d = 0  
        
        # Try differencing up to max_differencing times
        while d < max_differencing:
       
            adf, p_val, lag, nobs, cv, _ = adfuller(time_series.dropna())

            # If stationary: stop differencing
            if p_val < threshold:
                break

            time_series = time_series.diff()
            d += 1

        col_name = f"{col}_d{d}"
        stationary_df[col_name] = time_series.fillna(0)


    return stationary_df