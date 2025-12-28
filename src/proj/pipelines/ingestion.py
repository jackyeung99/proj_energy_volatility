from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from datetime import datetime

from proj.data.ingestion_state import get_last_available_date, compute_fetch_window, update_state
from proj.utils.paths import find_project_root, build_paths  # or your resolver

from proj.data.sources import yfin, fred, weather


@dataclass(frozen=True)
class SourceAdapter:
    name: str
    fetch: Callable[[datetime, datetime, dict], "object"]          # -> pd.DataFrame
    standardize: Callable[["object", dict], "object"]              # -> pd.DataFrame
    validate: Callable[["object", dict], None]


ADAPTERS: Dict[str, SourceAdapter] = {
    "equities": SourceAdapter("equities", yfin.fetch, yfin.standardize, yfin.validate),
    "macro":    SourceAdapter("macro",    fred.fetch,    fred.standardize,    fred.validate),
    "weather":  SourceAdapter("weather",  weather.fetch,  weather.standardize,  weather.validate),
}


def ingest_one_source(global_cfg: dict, base_ingest_cfg: dict, source_name: str, source_cfg: dict) -> dict:
    adapter = ADAPTERS[source_name]

    # 1) storage path (resolve relative paths once)
    store_path = resolve_from_project_root(global_cfg, source_cfg["store_path"])
    date_col = base_ingest_cfg["date_column"]

    # 2) last date
    last_date = get_last_available_date(store_path, date_col)

    # 3) compute window
    fetch_start, fetch_end = compute_fetch_window(
        last_date=last_date,
        lookback_days=base_ingest_cfg.get("lookback_days", 0),
        mode=base_ingest_cfg["mode"],
    )

    # 4) fetch
    new_df = adapter.fetch(fetch_start, fetch_end, source_cfg)

    # 5) standardize + validate
    new_df = adapter.standardize(new_df, source_cfg)
    adapter.validate(new_df, source_cfg)

    # 6) merge (per source)
    if store_path.exists():
        old_df = load_parquet(store_path)
        merged = merge_and_dedup(old_df, new_df, key=date_col)
    else:
        merged = new_df

    # 7) write
    atomic_write_parquet(merged, store_path)

    # 8) update state (optional)
    update_state(store_path, merged, {"ingestion": base_ingest_cfg, "source": source_cfg})

    # 9) return stats
    last_ts = merged[date_col].max() if len(merged) else None
    return {
        "source": source_name,
        "store_path": str(store_path),
        "rows_written": int(len(merged)),
        "last_timestamp": last_ts,
        "fetch_window": (fetch_start, fetch_end),
    }


def ingest(global_cfg: dict, step_cfg: dict) -> dict:
    base_ingest_cfg = step_cfg["ingestion"]
    sources_cfg = step_cfg["sources"]

    results = {}
    for source_name, cfg in sources_cfg.items():
        if not cfg.get("enabled", True):
            continue
        if source_name not in ADAPTERS:
            raise ValueError(f"Unknown source '{source_name}'. Known: {list(ADAPTERS)}")
        results[source_name] = ingest_one_source(global_cfg, base_ingest_cfg, source_name, cfg)

    return {"ingestion_results": results}
