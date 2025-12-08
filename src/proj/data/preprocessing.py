import pandas as pd
from sklearn.preprocessing import StandardScaler

from proj.data.reshaping import long_to_wide
from proj.features import transforms

def clean_stock_df(df):
    return long_to_wide(df)


def clean_macro_series(df):
    return df.reset_index(drop=True)


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
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(combined[exog_processed_cols])
    X = pd.DataFrame(X_scaled,
                     index=combined.index,
                     columns=exog_processed_cols)

    X = combined[exog_processed_cols]
    
    y = combined[target_cols]

    return X, y, scaler