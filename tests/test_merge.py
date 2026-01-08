import pandas as pd
from datetime import time

from proj.data.merge_helpers import merge_to_gold  
from proj.utils.dates import date_index_to_et_close_utc


class DummySpec:
    anchor_name = "equities"
    prefixes = {"macro": "macro_"}
    join_how = "left"
    start_date = None
    end_date = None
    close_et = time(16, 0)


def test_merge_to_gold_aligns_on_et_close_utc():
    # anchor: already ET-close UTC
    anchor = pd.DataFrame(
        {"rv_xle": [1.0, 2.0]},
        index=pd.to_datetime(["2026-01-07", "2026-01-08"]),
    )
    anchor = date_index_to_et_close_utc(anchor, close_et=time(16, 0))

    # macro: date-labeled daily
    macro = pd.DataFrame(
        {"VIXCLS": [10.0, 11.0]},
        index=pd.to_datetime(["2026-01-07", "2026-01-08"]),
    )

    out = merge_to_gold({"equities": anchor, "macro": macro}, DummySpec())

    assert len(out) == 2
    assert out.index.tz is not None
    # macro column should be prefixed
    assert "macro_VIXCLS" in out.columns
    # no drift: rows should match on same index values
    assert out["macro_VIXCLS"].tolist() == [10.0, 11.0]
