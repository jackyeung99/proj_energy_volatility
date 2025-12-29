from __future__ import annotations
import pandas as pd

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
