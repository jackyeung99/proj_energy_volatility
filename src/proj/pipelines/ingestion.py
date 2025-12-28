from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict
from datetime import datetime

from proj.data.ingestion_state import get_last_available_date, compute_fetch_window, update_state
from proj.data.sources import yfin, fred, weather

# NOTE: storage is injected (LocalStorage or URIStorage)
from proj.data.storage import Storage


@dataclass(frozen=True)
class SourceAdapter:
    name: str
    fetch: Callable[[datetime, datetime, dict], "object"]          # -> pd.DataFrame
    standardize: Callable[["object", dict], "object"]              # -> pd.DataFrame
    validate: Callable[["object", dict], None]


ADAPTERS: Dict[str, SourceAdapter] = {
    "equities": SourceAdapter("equities", yfin.fetch, yfin.standardize, yfin.validate),
    "macro":    SourceAdapter("macro",    fred.fetch, fred.standardize, fred.validate),
    "weather":  SourceAdapter("weather",  weather.fetch, weather.standardize, weather.validate),
}


def ingest_one_source(
    storage: Storage,
    global_cfg: dict,
    base_ingest_cfg: dict,
    source_name: str,
    source_cfg: dict,
) -> dict:
    adapter = ADAPTERS[source_name]

    # 1) storage key 
    store_key = source_cfg["store_path"]        
    date_col = base_ingest_cfg["date_column"]

    # 2) last date (teach get_last_available_date to accept storage+key)
    last_date = get_last_available_date(storage, store_key, date_col)

    # 3) compute window
    fetch_start, fetch_end = compute_fetch_window(
        last_date=last_date,
        lookback_days=base_ingest_cfg.get("lookback_days", 0),
        mode=base_ingest_cfg["mode"],
    )

    # 4) fetch
    new_df = adapter.fetch(fetch_start, fetch_end, source_cfg)

    # # 5) standardize + validate
    # new_df = adapter.standardize(new_df, source_cfg)
    # adapter.validate(new_df, source_cfg)

    # # 6) merge (per source)
    # if storage.exists(store_key):
    #     old_df = storage.read_parquet(store_key)
    #     merged = merge_and_dedup(old_df, new_df, key=date_col)   # keep your existing function
    # else:
    #     merged = new_df

    # # 7) write
    # storage.write_parquet(merged, store_key)

    # # 8) update state (modify update_state similarly: update_state(storage, key, ...))
    # update_state(storage, store_key, merged, {"ingestion": base_ingest_cfg, "source": source_cfg})

    # # 9) return stats
    # last_ts = merged[date_col].max() if len(merged) else None
    # return {
    #     "source": source_name,
    #     "store_key": store_key,
    #     "rows_written": int(len(merged)),
    #     "last_timestamp": last_ts,
    #     "fetch_window": (fetch_start, fetch_end),
    # }


def ingest(storage: Storage, global_cfg: dict, step_cfg: dict) -> dict:
    base_ingest_cfg = step_cfg["ingestion"]
    sources_cfg = step_cfg["sources"]

    results = {}
    for source_name, cfg in sources_cfg.items():
        if not cfg.get("enabled", True):
            continue
        if source_name not in ADAPTERS:
            raise ValueError(f"Unknown source '{source_name}'. Known: {list(ADAPTERS)}")
        results[source_name] = ingest_one_source(storage, global_cfg, base_ingest_cfg, source_name, cfg)

    return {"ingestion_results": results}
