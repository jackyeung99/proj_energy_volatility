
from datetime import datetime, timedelta, timezone
from typing import Optional

def get_last_available_date():

    pass




def compute_fetch_window(
    last_date: Optional[datetime],
    lookback_days: int,
    mode: str,
):
    """
    Compute the [fetch_start, fetch_end] window for ingestion.
    """

    fetch_end = datetime.now(timezone.utc).date()

    if mode == "full":
        if last_date is None:
            raise ValueError("Full mode requires a known start date or last_date.")
        fetch_start = last_date - timedelta(days=lookback_days)

    elif mode == "incremental":
        if last_date is None:
            raise ValueError("Incremental mode requires last_date.")
        fetch_start = last_date - timedelta(days=lookback_days)

    else:
        raise ValueError(f"Unknown ingestion mode: {mode}")

    return fetch_start, fetch_end


def update_state():


    pass