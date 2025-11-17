from pandas_datareader import data as pdr

def fetch_fred_series(features, start, end):

    
    fred = pdr.DataReader(
        features,
        "fred",
        start=start,
        end=end
    ).astype(float).reset_index()

    return fred.ffill()

