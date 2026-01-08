# src/proj/utils/dates.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo
import pandas as pd

ET = ZoneInfo("America/New_York")


# -----------------------
# IDs / index utilities
# -----------------------
def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def ensure_datetime_index_utc(df: pd.DataFrame, *, allow_naive: bool = True) -> pd.DataFrame:
    """
    Ensure df has a sorted, timezone-aware UTC DatetimeIndex.

    - If index is tz-naive and allow_naive=True, interpret it as UTC and localize.
    - If index is tz-aware, convert to UTC.
    """
    out = df.copy().sort_index()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("dates.ensure_datetime_index_utc: index must be a DatetimeIndex")

    idx = out.index

    if idx.tz is None:
        if not allow_naive:
            raise ValueError("dates.ensure_datetime_index_utc: tz-naive index not allowed")
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    out.index = idx
    return out


# -----------------------
# Trading-day logic (weekday-only)
# -----------------------
def is_business_day_et(day_et: pd.Timestamp) -> bool:
    """Weekday-only business day check (Mon-Fri)."""
    d = pd.Timestamp(day_et).normalize()
    return d.dayofweek < 5


def prev_business_day_et(day_et: pd.Timestamp) -> pd.Timestamp:
    """Previous weekday business day label in ET."""
    d = pd.Timestamp(day_et).normalize() - pd.Timedelta(days=1)
    while not is_business_day_et(d):
        d -= pd.Timedelta(days=1)
    return d.normalize()


def next_business_day_et(day_et: pd.Timestamp) -> pd.Timestamp:
    """Next weekday business day label in ET."""
    d = pd.Timestamp(day_et).normalize() + pd.Timedelta(days=1)
    while not is_business_day_et(d):
        d += pd.Timedelta(days=1)
    return d.normalize()


# -----------------------
# ET close <-> UTC identity timestamps
# -----------------------
def et_close_ts_utc(day_et: pd.Timestamp, close_et: time = time(16, 0)) -> pd.Timestamp:
    """
    Return the UTC timestamp corresponding to ET market close on the ET calendar day.

    This is the recommended daily identity key for both gold and forecast targets.
    DST-safe (because we localize to America/New_York then convert to UTC).
    """
    day_et = pd.Timestamp(day_et).normalize()
    ts_et = (
        day_et.tz_localize(ET)
        .replace(hour=close_et.hour, minute=close_et.minute, second=0, microsecond=0)
    )
    return ts_et.tz_convert("UTC")


def resolve_asof_trade_date_et(
    now_utc: datetime,
    close_et: time = time(16, 0),
) -> pd.Timestamp:
    """
    Decide the last fully-known trading day label in ET.

    Rules (weekday-only):
      - If today is Sat/Sun -> asof = previous business day
      - If weekday AND before close -> asof = previous business day
      - If weekday AND after close -> asof = today
    """
    if now_utc.tzinfo is None:
        raise ValueError("resolve_asof_trade_date_et: now_utc must be timezone-aware UTC datetime")

    now_et = now_utc.astimezone(ET)
    today_et = pd.Timestamp(now_et.date()).normalize()

    if not is_business_day_et(today_et):
        return prev_business_day_et(today_et)

    if now_et.time() < close_et:
        return prev_business_day_et(today_et)

    return today_et


# -----------------------
# Normalizers for gold construction
# -----------------------
def normalize_to_et_close_utc(df: pd.DataFrame, close_et: time = time(16, 0)) -> pd.DataFrame:
    """
    Normalize a tz-aware timestamp index to ET close (converted to UTC).

    Use this when you already have timestamped rows (intraday or daily) but want a
    stable daily identity key (ET close in UTC).
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("normalize_to_et_close_utc: index must be a DatetimeIndex")

    idx = out.index
    if idx.tz is None:
        raise ValueError("normalize_to_et_close_utc: index must be tz-aware (UTC recommended)")

    idx_et = idx.tz_convert(ET)
    idx_et_close = idx_et.map(
        lambda ts: ts.replace(
            hour=close_et.hour,
            minute=close_et.minute,
            second=0,
            microsecond=0,
        )
    )
    out.index = idx_et_close.tz_convert("UTC")
    return out


def date_index_to_et_close_utc(df: pd.DataFrame, close_et: time = time(16, 0)) -> pd.DataFrame:
    """
    Convert an index representing *ET calendar dates* into the UTC timestamp of
    ET market close for those dates.

    Accepts:
      - DATE / string dates (tz-naive)
      - tz-naive datetimes
      - tz-aware datetimes (will be converted to ET first, then date-extracted)

    Output:
      - tz-aware UTC DatetimeIndex at ET close (DST-safe)
    """
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        # allow strings/dates etc.
        idx = pd.to_datetime(out.index, errors="raise")
    else:
        idx = out.index

    # If tz-aware, convert to ET and take the ET date label (normalize in ET).
    # If tz-naive, treat it as an ET date label directly.
    if idx.tz is not None:
        idx_et_dates = idx.tz_convert(ET).normalize()
    else:
        idx_et_dates = pd.to_datetime(idx).normalize().tz_localize(ET)

    # Stamp to ET close on that ET date
    idx_et_close = idx_et_dates.map(
        lambda ts: ts.replace(
            hour=close_et.hour,
            minute=close_et.minute,
            second=0,
            microsecond=0,
        )
    )

    out.index = idx_et_close.tz_convert("UTC")
    return out



# -----------------------
# Convenience: forecast target identity
# -----------------------
def forecast_target_ts_utc(
    asof_trade_date_et: pd.Timestamp,
    horizon_bdays: int = 1,
    close_et: time = time(16, 0),
) -> pd.Timestamp:
    """
    Compute the UTC identity timestamp for a forecast target:
      - advance horizon_bdays ET business days from asof_trade_date_et
      - map that ET date to ET close, converted to UTC
    """
    if horizon_bdays < 1:
        raise ValueError("forecast_target_ts_utc: horizon_bdays must be >= 1")

    d = pd.Timestamp(asof_trade_date_et).normalize()
    for _ in range(horizon_bdays):
        d = next_business_day_et(d)
    return et_close_ts_utc(d, close_et=close_et)



def duration_to_dates(duration: str, end=None):
    """
    Convert IBKR duration strings like '2 Y', '30 D', '3 M', '12 W'
    into (start_date, end_date).
    """
    # end date defaults to NOW unless provided
    if end is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.fromisoformat(end)

    value, unit = duration.split()
    value = int(value)
    unit = unit.upper()

    if unit.startswith("D"):
        start_dt = end_dt - timedelta(days=value)
    elif unit.startswith("W"):
        start_dt = end_dt - timedelta(weeks=value)
    elif unit.startswith("M"):
        start_dt = end_dt - relativedelta(months=value)
    elif unit.startswith("Y"):
        start_dt = end_dt - relativedelta(years=value)
    else:
        raise ValueError(f"Unsupported duration unit: {unit}")

    return start_dt.date(), end_dt.date()



def chunk_dates(duration, chunk_size):
    """
    Split a date range into smaller contiguous chunks.
    """
    start, end = map(pd.to_datetime, duration)

    chunks = []
    current = start

    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_size - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)

    return chunks


