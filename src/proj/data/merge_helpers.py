from __future__ import annotations

import pandas as pd
from datetime import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from proj.utils.dates import normalize_to_et_close_utc, ensure_datetime_index_utc, date_index_to_et_close_utc



def standardize_daily_identity_index(
    df: pd.DataFrame,
    name: str,
    close_et: time = time(16, 0),
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: df must be indexed by a DatetimeIndex")

    df = df.copy().sort_index()

    if df.index.tz is not None:
        # tz-aware timestamps: snap to ET close then convert to UTC
        return normalize_to_et_close_utc(df, close_et=close_et)

    # tz-naive daily labels: interpret as ET dates, then stamp to ET close -> UTC
    return date_index_to_et_close_utc(df, close_et=close_et)

def _prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out

def _prefix_cols_except(df: pd.DataFrame, prefix: str, exclude: set[str]) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c in exclude:
            rename[c] = c
        else:
            rename[c] = f"{prefix}{c}"
    return df.rename(columns=rename)


def _validate_lags_only(df: pd.DataFrame, name: str, allow: Optional[Sequence[str]] = None) -> None:
    """
    For exogenous datasets, enforce lags-only columns to prevent leakage.
    """
    allow = set(allow or [])
    bad = []
    for c in df.columns:
        if c in allow:
            continue
        if "_lag" not in c:
            bad.append(c)
    if bad:
        raise ValueError(
            f"{name}: found non-lagged columns (potential leakage): {bad[:10]}"
            + (" ..." if len(bad) > 10 else "")
        )


@dataclass
class MergeSpec:
    anchor_name: str = "equities"
    join_how: str = "left"  # left keeps anchor dates
    dropna_target: bool = True
    target_cols: Tuple[str, ...] = ("log_rv_idio",)
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Prefixes for non-anchor datasets
    prefixes: Optional[Dict[str, str]] = None

    # Enforce lags-only on non-anchor datasets
    enforce_lags_only: bool = True



def standardize_daily_identity_index(
    df: pd.DataFrame,
    name: str,
    close_et: time = time(16, 0),
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: df must be indexed by a DatetimeIndex")

    df = df.copy().sort_index()

    if df.index.tz is not None:
        # tz-aware timestamps: snap to ET close then convert to UTC
        return normalize_to_et_close_utc(df, close_et=close_et)

    # tz-naive daily labels: interpret as ET dates, then stamp to ET close -> UTC
    return date_index_to_et_close_utc(df, close_et=close_et)


def merge_to_gold(datasets: Dict[str, pd.DataFrame], spec: MergeSpec) -> pd.DataFrame:
    if spec.anchor_name not in datasets:
        raise ValueError(f"merge_to_gold: missing anchor dataset '{spec.anchor_name}'")

    prefixes = spec.prefixes or {
        "equities_daily": "etf_",
        "macro": "macro_",
        "weather": "wx_",
    }

    # columns that should NEVER be prefixed
    no_prefix_cols = {"XLE", "SPY"}

    close_et = getattr(spec, "close_et", time(16, 0))

    # 1) Anchor
    anchor = standardize_daily_identity_index(
        datasets[spec.anchor_name],
        spec.anchor_name,
        close_et=close_et,
    )

    if spec.start_date is not None:
        anchor = anchor.loc[anchor.index >= pd.Timestamp(spec.start_date, tz="UTC")]
    if spec.end_date is not None:
        anchor = anchor.loc[anchor.index <= pd.Timestamp(spec.end_date, tz="UTC")]

    out = anchor.copy()

    # 2) Merge remaining datasets
    for name, df in datasets.items():
        if name == spec.anchor_name:
            continue

        df = standardize_daily_identity_index(df, name, close_et=close_et)

        pref = prefixes.get(name, f"{name}_")

        if name == "equities_daily":
            df = _prefix_cols_except(df, pref, exclude=no_prefix_cols)
        else:
            df = _prefix_cols(df, pref)

        out = out.join(df, how=spec.join_how)

    return out


def merge_and_dedup(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    if old_df is None or old_df.empty:
        return new_df.copy()
    if new_df is None or new_df.empty:
        return old_df.copy()

    old = old_df.copy()
    new = new_df.copy()

    # Ensure datetime index (UTC-safe)
    old.index = pd.to_datetime(old.index, utc=True, errors="coerce")
    new.index = pd.to_datetime(new.index, utc=True, errors="coerce")

    old = old[~old.index.isna()]
    new = new[~new.index.isna()]

    # If there are duplicates inside either df, keep last within each
    old = old[~old.index.duplicated(keep="last")]
    new = new[~new.index.duplicated(keep="last")]

    # Concatenate; keep="last" means NEW wins if we put it last
    merged = pd.concat([old, new], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]

    # Sort by time
    merged = merged.sort_index()

    return merged

import pandas as pd

def merge_and_dedup_long(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    long_col: str,
    *,
    keep: str = "last",
    sort: bool = True,
) -> pd.DataFrame:
    """
    Merge two long-form DataFrames (time-series index) and de-duplicate on (index, long_col).

    Assumes:
      - index is datetime-like (will be normalized to UTC)
      - long_col exists (e.g., 'model')
    """

    # Handle empty inputs
    if old_df is None or len(old_df) == 0:
        out = new_df.copy()
    elif new_df is None or len(new_df) == 0:
        out = old_df.copy()
    else:
        out = pd.concat([old_df, new_df], axis=0)

    if out is None or len(out) == 0:
        return out

    # Validate
    if long_col not in out.columns:
        raise ValueError(f"merge_and_dedup_long: missing required column '{long_col}'")

    # Normalize index to tz-aware UTC
    out = out.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.loc[~out.index.isna()]
    out[long_col] = out[long_col].astype(str)

    # dedup on (time index, model)
    tmp = out.reset_index(names="ts")
    tmp = tmp.drop_duplicates(subset=["ts", long_col], keep=keep)

    tmp = tmp.sort_values(["ts", long_col])
    tmp = tmp.set_index("ts")
    return tmp
