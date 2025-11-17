import pandas as pd

from proj.data.reshaping import long_to_wide

def clean_stock_df(df):
    return long_to_wide(df)


def clean_macro_series(df):
    return df.reset_index(drop=True)