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

def parse_hhmm(s: str, default: str = "16:00") -> time:
    if not s:
        s = default
    hh, mm = s.split(":")
    return time(int(hh), int(mm))

def ensure_datetime_index_utc(df: pd.DataFrame, *, name: str = "df") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: index must be a DatetimeIndex")

    out = df.copy()
    idx = out.index

    if idx.tz is None:
        out.index = pd.to_datetime(idx).tz_localize("UTC")
    else:
        out.index = idx.tz_convert("UTC")

    return out.sort_index()

def is_midnight_utc_labels(idx: pd.DatetimeIndex) -> bool:
    """
    True if timestamps are exactly midnight when viewed in UTC (robust to tz implementations).
    """
    if idx.tz is None:
        return False
    idx_utc = idx.tz_convert("UTC")
    return (idx_utc == idx_utc.normalize()).all()

def labels_as_et_close_utc_from_utc_midnight(idx: pd.DatetimeIndex, close_et: time) -> pd.DatetimeIndex:
    """
    idx are tz-aware and represent daily labels at midnight UTC.
    Interpret the *date label* (UTC date) then stamp ET close for that date.
    """
    idx_utc = idx.tz_convert("UTC")
    dates = idx_utc.normalize().tz_localize(None)          # tz-naive dates
    et_midnight = dates.tz_localize(ET)                    # localize as ET-midnight of the label date
    et_close = et_midnight + pd.Timedelta(hours=close_et.hour, minutes=close_et.minute)
    return et_close.tz_convert("UTC")

def labels_as_et_close_utc_from_naive_dates(idx: pd.DatetimeIndex, close_et: time) -> pd.DatetimeIndex:
    """
    idx are tz-naive date labels (YYYY-MM-DD). Interpret as ET date labels then stamp ET close.
    """
    et_midnight = pd.to_datetime(idx).normalize().tz_localize(ET)
    et_close = et_midnight + pd.Timedelta(hours=close_et.hour, minutes=close_et.minute)
    return et_close.tz_convert("UTC")

def instants_to_et_close_utc(idx: pd.DatetimeIndex, close_et: time) -> pd.DatetimeIndex:
    """
    idx are real instants. Map each instant to its ET calendar day, then stamp ET close.
    """
    et = idx.tz_convert(ET)
    et_close = et.normalize() + pd.Timedelta(hours=close_et.hour, minutes=close_et.minute)
    return et_close.tz_convert("UTC")

def standardize_daily_identity_index(
    df: pd.DataFrame,
    *,
    close_et: time = time(16, 0),
    name: str = "df",
) -> pd.DataFrame:
    """
    Normalize index to "ET close expressed in UTC" without snap-back.

    - tz-naive => ET date labels
    - tz-aware midnight UTC => UTC date labels
    - tz-aware non-midnight => real instants
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: df must have a DatetimeIndex")

    out = df.copy().sort_index()
    idx = out.index

    if idx.tz is None:
        out.index = labels_as_et_close_utc_from_naive_dates(idx, close_et)
        return out

    if is_midnight_utc_labels(idx):
        out.index = labels_as_et_close_utc_from_utc_midnight(idx, close_et)
        return out

    out.index = instants_to_et_close_utc(idx, close_et)
    return out

def filter_rth(
    df: pd.DataFrame,
    *,
    start: str = "09:30",
    end: str = "16:00",
    tz: ZoneInfo = ET,
    name: str = "df",
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError(f"{name}: filter_rth expects tz-aware DatetimeIndex")

    et = df.tz_convert(tz)
    start_t = pd.Timestamp(start).time()
    end_t = pd.Timestamp(end).time()
    mask = (
        (et.index.dayofweek < 5)
        & (et.index.time >= start_t)
        & (et.index.time <= end_t)
    )
    return df.loc[mask]


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


