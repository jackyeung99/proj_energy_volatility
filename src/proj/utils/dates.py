# src/proj/utils/dates.py
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd

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


