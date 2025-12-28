from __future__ import annotations

import yfinance as yf 
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, List, Optional, Tuple, Union, Dict
from proj.utils.dates import chunk_dates

import pandas as pd


DateLike = Union[str, pd.Timestamp]


@dataclass(frozen=True)
class FetchConfig:
    """
    Configuration for fetching and chunking.

    tickers: list of tickers to fetch
    interval: data granularity (depends on your provider): "1d", "1h", "5m", etc.
    chunk_size_days: chunk date ranges into this many days per request
    """
    tickers: List[str]
    interval: str = "1d"
    chunk_size_days: int = 7


def fetch(
    features: FetchConfig,
    start: DateLike,
    end: DateLike
) -> pd.DataFrame:
    """
    Fetch stock data using chunked date ranges.

    Parameters
    ----------
    features : FetchConfig
        What to fetch (tickers/interval/chunk size)
    start, end : DateLike
        Inclusive date bounds for fetching
    fetch_impl : callable
        A provider-specific function you plug in. Signature:
            fetch_impl(ticker: str, start: pd.Timestamp, end: pd.Timestamp, interval: str) -> pd.DataFrame
        It must return a DataFrame with at least:
            - a datetime column named 'timestamp' OR a DatetimeIndex
            - numeric columns like 'open','high','low','close','volume' (optional)

    Returns
    -------
    pd.DataFrame
        Long-format data with columns: ['ticker','timestamp', ...]
    """


    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    chunks = chunk_dates((start_ts, end_ts), features.chunk_size_days)
    out: List[pd.DataFrame] = []

    for ticker in features.tickers:
        for cstart, cend in chunks:
            df = yf.Ticker(ticker=ticker, start=cstart, end=cend, interval=features.interval)

            if df is None or len(df) == 0:
                continue

            df = df.copy()

            # Normalize timestamp
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
            elif isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.columns[0] if df.columns.size else "index": "timestamp"})
                # If reset_index didn't name properly, ensure timestamp exists
                if "timestamp" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "timestamp"})
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
            else:
                raise ValueError("fetch_impl must return a DataFrame with 'timestamp' column or a DatetimeIndex.")

            df["ticker"] = ticker

            # Keep only chunk window (defensive)
            df = df[(df["timestamp"] >= cstart) & (df["timestamp"] <= (cend + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))]

            out.append(df)

    if not out:
        return pd.DataFrame(columns=["ticker", "timestamp"])

    return pd.concat(out, ignore_index=True)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names/dtypes and create basic derived fields.

    Output contract:
      - columns include: ticker (str), timestamp (datetime64), close (float) if available
      - sorted by (ticker, timestamp)
      - duplicates dropped
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "timestamp"])

    x = df.copy()

    # Common column-name normalization
    rename_map = {}
    for col in x.columns:
        lc = col.lower()
        if lc in {"datetime", "date", "time", "ts"} and "timestamp" not in x.columns:
            rename_map[col] = "timestamp"
        elif lc in {"adj close", "adj_close", "adjclose"}:
            rename_map[col] = "adj_close"
        elif lc in {"close"}:
            rename_map[col] = "close"
        elif lc in {"open"}:
            rename_map[col] = "open"
        elif lc in {"high"}:
            rename_map[col] = "high"
        elif lc in {"low"}:
            rename_map[col] = "low"
        elif lc in {"volume", "vol"}:
            rename_map[col] = "volume"

    if rename_map:
        x = x.rename(columns=rename_map)

    if "ticker" not in x.columns or "timestamp" not in x.columns:
        raise ValueError("Expected at least columns ['ticker','timestamp'] after standardization.")

    x["ticker"] = x["ticker"].astype(str)
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")

    # Drop rows with invalid timestamps
    x = x.dropna(subset=["timestamp"])

    # Enforce numeric types if present
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    # Sort and dedupe
    x = x.sort_values(["ticker", "timestamp"])
    x = x.drop_duplicates(subset=["ticker", "timestamp"], keep="last").reset_index(drop=True)

    # Optional: compute log returns if close is present
    if "close" in x.columns:
        x["log_return"] = (
            x.groupby("ticker")["close"]
            .apply(lambda s: (s.astype(float)).replace(0, pd.NA))
            .groupby(level=0)
            .apply(lambda s: (s / s.shift(1)).apply(lambda v: pd.NA if pd.isna(v) else float(v)))
        )
        # The above is conservative; alternatively:
        # x["log_return"] = x.groupby("ticker")["close"].apply(lambda s: np.log(s).diff()).reset_index(level=0, drop=True)

        # Clean non-finite returns
        x["log_return"] = pd.to_numeric(x["log_return"], errors="coerce")

    return x


def validate(
    df: pd.DataFrame,
    require_ohlc: bool = False,
    min_rows_per_ticker: int = 2
) -> Dict[str, object]:
    """
    Validate standardized stock data.

    Checks:
      - required columns exist
      - no null timestamps
      - uniqueness of (ticker, timestamp)
      - monotonic ordering within ticker
      - optional OHLC presence
      - minimum rows per ticker

    Returns a small report dict. Raises ValueError on failure.
    """
    if df is None or df.empty:
        raise ValueError("No data to validate (empty DataFrame).")

    required = {"ticker", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["timestamp"].isna().any():
        raise ValueError("Found NaN timestamps after standardization.")

    # Uniqueness
    dupes = df.duplicated(subset=["ticker", "timestamp"]).sum()
    if dupes > 0:
        raise ValueError(f"Found {dupes} duplicate (ticker,timestamp) rows.")

    # Monotonic within ticker
    bad = 0
    for t, g in df.groupby("ticker", sort=False):
        if not g["timestamp"].is_monotonic_increasing:
            bad += 1
    if bad > 0:
        raise ValueError(f"{bad} tickers have non-monotonic timestamps.")

    # Optional OHLC requirement
    if require_ohlc:
        for c in ["open", "high", "low", "close"]:
            if c not in df.columns:
                raise ValueError(f"require_ohlc=True but '{c}' is missing.")
            if df[c].isna().all():
                raise ValueError(f"Column '{c}' exists but is all NaN.")

    # Minimum rows
    counts = df.groupby("ticker")["timestamp"].size()
    too_small = counts[counts < min_rows_per_ticker]
    if len(too_small) > 0:
        raise ValueError(f"Not enough rows for tickers: {too_small.to_dict()}")

    report = {
        "tickers": counts.index.tolist(),
        "rows": int(len(df)),
        "min_rows_per_ticker": int(counts.min()),
        "max_rows_per_ticker": int(counts.max()),
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "columns": df.columns.tolist(),
    }
    return report
