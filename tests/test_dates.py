import pandas as pd
import pytest
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from proj.utils.dates import (
    et_close_ts_utc,
    normalize_to_et_close_utc,
    date_index_to_et_close_utc,
    resolve_asof_trade_date_et,
    next_business_day_et,
)

ET = ZoneInfo("America/New_York")


def _assert_et_close_identity(idx: pd.DatetimeIndex, close_et=time(16, 0)):
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.tz is not None
    assert str(idx.tz) in ("UTC", "UTC+00:00")

    idx_et = idx.tz_convert(ET)
    assert (idx_et.hour == close_et.hour).all()
    assert (idx_et.minute == close_et.minute).all()
    assert (idx_et.second == 0).all()
    assert (idx_et.microsecond == 0).all()


def test_et_close_ts_utc_is_dst_safe():
    # Winter date (EST): 16:00 ET == 21:00 UTC
    d_w = pd.Timestamp("2026-01-07")
    ts_w = et_close_ts_utc(d_w, close_et=time(16, 0))
    assert ts_w.tz is not None
    assert ts_w.tz_convert(ET).hour == 16
    assert ts_w.hour in (20, 21)  # depends on DST (winter should be 21)

    # Summer date (EDT): 16:00 ET == 20:00 UTC
    d_s = pd.Timestamp("2026-07-07")
    ts_s = et_close_ts_utc(d_s, close_et=time(16, 0))
    assert ts_s.tz_convert(ET).hour == 16
    assert ts_s.hour in (20, 21)  # summer should be 20


def test_normalize_to_et_close_utc_snaps_same_day():
    idx = pd.to_datetime(
        ["2026-01-07T15:00:00Z", "2026-01-07T23:30:00Z"]
    ).tz_convert("UTC")
    df = pd.DataFrame({"x": [1, 2]}, index=idx)

    out = normalize_to_et_close_utc(df, close_et=time(16, 0))
    _assert_et_close_identity(out.index)

    # Both should map to same ET day close → duplicate index is expected
    assert out.index[0] == out.index[1]


def test_date_index_to_et_close_utc_treats_dates_as_et():
    df = pd.DataFrame({"x": [1, 2]}, index=pd.to_datetime(["2026-01-07", "2026-01-08"]))
    out = date_index_to_et_close_utc(df, close_et=time(16, 0))

    _assert_et_close_identity(out.index)
    # Should be in UTC close timestamps, not midnight
    assert (out.index.hour != 0).all()


@pytest.mark.parametrize(
    "now_utc, expected_asof",
    [
        # Before close on a weekday → previous business day
        (datetime(2026, 1, 7, 15, 0, tzinfo=timezone.utc), pd.Timestamp("2026-01-06")),
        # After close on a weekday → same day
        (datetime(2026, 1, 7, 23, 0, tzinfo=timezone.utc), pd.Timestamp("2026-01-07")),
        # Weekend → previous business day (Fri)
        (datetime(2026, 1, 10, 18, 0, tzinfo=timezone.utc), pd.Timestamp("2026-01-09")),
    ],
)
def test_resolve_asof_trade_date_et(now_utc, expected_asof):
    asof = resolve_asof_trade_date_et(now_utc, close_et=time(16, 0))
    assert asof == expected_asof.normalize()


def test_next_business_day_et_skips_weekend():
    fri = pd.Timestamp("2026-01-09")  # Friday
    nxt = next_business_day_et(fri)
    assert nxt == pd.Timestamp("2026-01-12")  # Monday
