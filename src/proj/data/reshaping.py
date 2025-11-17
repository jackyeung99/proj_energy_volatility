import pandas as pd

def long_to_wide(df, pivot_by='close'):

    wide = df.pivot(index='date', columns='Symbol', values=pivot_by)
    wide = wide.reset_index()  
    wide.columns.name = None

    return wide