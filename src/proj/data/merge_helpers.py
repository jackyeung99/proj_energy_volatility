from __future__ import annotations
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple



def _ensure_dt_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{name}: dataframe is empty or None")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: must have DatetimeIndex")
    return df.sort_index()


def _normalize_daily_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize index to midnight UTC day boundaries. Keeps DatetimeIndex type.
    """
    out = df.copy()
    idx = out.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    out.index = idx.normalize()
    return out


def _prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


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


def merge_to_gold(datasets: Dict[str, pd.DataFrame], spec: MergeSpec) -> pd.DataFrame:
    if spec.anchor_name not in datasets:
        raise ValueError(f"merge_to_gold: missing anchor dataset '{spec.anchor_name}'")

    prefixes = spec.prefixes or {
        "equities_daily": "etf_",
        "macro": "macro_",
        "weather": "wx_",
    }

    # Anchor
    anchor = _ensure_dt_index(datasets[spec.anchor_name], spec.anchor_name)
    anchor = _normalize_daily_index_utc(anchor)

    if spec.start_date is not None:
        anchor = anchor.loc[anchor.index >= pd.Timestamp(spec.start_date)]
    if spec.end_date is not None:
        anchor = anchor.loc[anchor.index <= pd.Timestamp(spec.end_date)]

    out = anchor.copy()

    # Merge others
    for name, df in datasets.items():
        if name == spec.anchor_name:
            continue

        df = _ensure_dt_index(df, name)
        df = _normalize_daily_index_utc(df)

        if spec.enforce_lags_only:
            _validate_lags_only(df, name)

        pref = prefixes.get(name, f"{name}_")
        df = _prefix_cols(df, pref)

        out = out.join(df, how=spec.join_how)

    # Require target(s)
    if spec.dropna_target and spec.target_cols:
        missing = [c for c in spec.target_cols if c not in out.columns]
        if missing:
            raise ValueError(f"merge_to_gold: missing required target columns: {missing}")
        out = out.dropna(subset=list(spec.target_cols))

    # Drop all-NA columns (happens when a dataset is shorter)
    out = out.dropna(axis=1, how="all")

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
