import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
from datetime import time

from proj.features.preprocessing import preprocess_equities  # adjust import
from proj.utils.dates import ET

ET = ZoneInfo("America/New_York")
def test_preprocess_equities_daily_index_is_et_close_utc():
    # Need >= window+1 rows to get first non-NaN XLE_idio
    window = 2
    n = window + 2  # a little extra

    # Intraday timestamps within the same ET trading day
    idx_et = pd.date_range("2026-01-07 10:00", periods=n, freq="30min", tz=ET)
    idx_utc = idx_et.tz_convert("UTC")

    # Make prices move so returns aren't all zero
    xle = 100 + np.cumsum(np.ones(n))     # 100,101,102,...
    spy = 400 + np.cumsum(np.ones(n))     # 400,401,402,...

    df = pd.DataFrame({"XLE": xle, "SPY": spy}, index=idx_utc)

    base_cfg = {"resample_freq": "1D"}
    src_cfg = {
        "idio_window": window,
        "min_intraday_bins": 1,
        "log_eps": 1e-12,
        "market_close_et": "16:00",
    }

    daily = preprocess_equities(df, base_cfg, src_cfg)

    # Should produce exactly one daily row
    assert len(daily) == 1

    # Index should be UTC and correspond to 16:00 ET
    assert daily.index.tz is not None
    idx_et_close = daily.index.tz_convert(ET)
    assert idx_et_close[0].hour == 16
    assert idx_et_close[0].minute == 0

    # Has expected columns
    for c in ["rv_xle", "rv_spy", "rv_idio", "n_intra"]:
        assert c in daily.columns
