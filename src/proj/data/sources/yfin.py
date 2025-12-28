from __future__ import annotations

import yfinance as yf 
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, List, Optional, Tuple, Union, Dict, Sequence
from proj.utils.dates import chunk_dates

import pandas as pd


DateLike = Union[str, pd.Timestamp]


def fetch(
    tickers: Union[str, Sequence[str]],
    start: DateLike,
    end: DateLike,
    interval: str = "1h",
    chunk_size_days: int = 60,
) -> pd.DataFrame:
    """
    Fetch multiple tickers using yfinance.Tickers.history, chunked by date.

    Returns a tidy DataFrame with columns: [timestamp, ticker, close].
    """

    # normalize tickers into a list
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split() if t.strip()]

    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    # treat date-only end as inclusive
    if end_ts == end_ts.normalize():
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # compute chunks
    chunks = chunk_dates((start_ts, end_ts), chunk_size=chunk_size_days)
    pieces: List[pd.DataFrame] = []

    # create the yfinance Tickers object once
    tick_obj = yf.Tickers(" ".join(tickers))

    for cstart, cend in chunks:
        # note: end is often exclusive in Yahoo — we add 1h buffer
        df = tick_obj.history(
            start=cstart,
            end=cend ,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if df is None or df.empty:
            continue

        # index is timestamp, level 0 of columns is ticker
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True)

        # restrict to exact chunk range
        df = df.loc[(df.index >= cstart) & (df.index <= cend)]

        # if multicolumn: (ticker, OHLC...)
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df.xs("Close", level=1, axis=1)
        else:
            # fallback if yfinance returns single-ticker flat columns
            close_df = df[["Close"]]
            close_df.columns = [tickers[0]]

        # stack tidy: timestamp | ticker | close
        tidy = close_df.stack().reset_index()
        tidy.columns = ["timestamp", "ticker", "close"]

        pieces.append(tidy)

    if not pieces:
        # return empty tidy structure if no data
        return pd.DataFrame(columns=["timestamp", "ticker", "close"])

    # combine chunks, dedupe overlaps
    out = (
        pd.concat(pieces, ignore_index=True)
          .sort_values(["ticker", "timestamp"])
          .drop_duplicates(["ticker", "timestamp"], keep="last")
          .reset_index(drop=True)
    )

    return out



def standardize(df: pd.DataFrame) -> pd.DataFrame:
    

    df = df.copy()

    # Ensure timestamp is datetime + UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    wide = (
        df.pivot_table(
            index="timestamp",
            columns="ticker",
            values="close",
            aggfunc="last",   # safe if duplicates exist
        )
        .sort_index()
    )

    return wide

def validate(df: pd.DataFrame) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("yfinance: df must be a DataFrame")
    if df.empty:
        raise ValueError("yfinance: DataFrame is empty")

    # Index checks
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("yfinance: index must be a DatetimeIndex (timestamp)")
    if df.index.hasnans:
        raise ValueError("yfinance: index contains NaNs")
    if df.index.duplicated().any():
        raise ValueError("yfinance: duplicate timestamps in index")
    if not df.index.is_monotonic_increasing:
        raise ValueError("yfinance: timestamps must be sorted increasing")

    # Column checks
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("yfinance: columns should not be MultiIndex after pivot")
    if df.shape[1] == 0:
        raise ValueError("yfinance: no ticker columns found")
    if df.columns.duplicated().any():
        raise ValueError("yfinance: duplicate ticker columns found")

    # Data checks
    if df.isna().all().any():
        bad = df.columns[df.isna().all()].tolist()
        raise ValueError(f"yfinance: tickers entirely NaN: {bad}")

    # Ensure numeric (close should be numeric)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"yfinance: non-numeric ticker columns: {non_numeric}")

    # Optional plausibility: prices should be > 0 whenever present
    if (df.dropna(how="all").le(0)).any().any():
        raise ValueError("yfinance: found non-positive prices")

if __name__ == "__main__":

    features = ["SPY", "XLE"]    
    start = "2025-12-12"
    end = "2025-12-20"

    df = fetch(features, start, end)
    df = standardize(df)

    validate(df)
    print(df.head())
