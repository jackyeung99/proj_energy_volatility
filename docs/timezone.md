# Timezone & Timestamp Logic Checklist (Gold + Predictions)

This checklist defines the **complete, consistent timezone logic** needed so
daily realized data (`gold`) and forecasts (`predictions`) always align,
including the extra future forecast row.

---

## A. Canonical Identity Standard

1. **Canonical timezone:** UTC only (storage + joins).
2. **Canonical daily identity:** ET market close (16:00 ET) converted to UTC.
3. All daily rows in **gold** and all forecast targets in **predictions**
   must use this same UTC timestamp as their identity.

---

## B. Gold Construction (from intraday data)

4. Ingest intraday data as **timezone-aware UTC timestamps**.
5. Convert UTC → ET **only for grouping** intraday data into trading days.
6. For each ET trading day:
   - Compute realized volatility from intraday returns.
   - Assign exactly **one identity timestamp**:
     - `timestamp_utc = (ET trade date @ 16:00 ET) → UTC`
7. Optionally keep label columns:
   - `trade_date_et` (DATE)
   - Do **not** use label columns for joins.

---

## C. Prediction Run Timing (script execution)

8. Record `run_time_utc = now()` (UTC).
9. Determine `asof_trade_date_et`:
   - Weekend → previous business day
   - Weekday **before close** → previous business day
   - Weekday **after close** → today
10. Compute:
    - `asof_close_ts_utc = (asof_trade_date_et @ 16:00 ET) → UTC`
11. Slice training data using:
    - `timestamp <= asof_close_ts_utc`
    - Do **not** slice using midnight boundaries.

---

## D. Forecast Target Timestamp (join key)

12. Compute `forecast_trade_date_et`:
    - For 1-step ahead: next business day after `asof_trade_date_et`.
13. Compute the forecast identity:
    - `forecast_ts_utc = (forecast_trade_date_et @ 16:00 ET) → UTC`
14. Store both:
    - `forecast_trade_date_et` (label)
    - `forecast_ts_utc` (identity / join key)

---

## E. Prediction Table Schema (required fields)

15. `predictions` must include:
    - `run_id`
    - `run_time_utc`
    - `asof_trade_date_et`
    - `asof_close_ts_utc`
    - `forecast_trade_date_et`
    - **`forecast_ts_utc`** (primary key)
    - model forecast columns (EWMA, GARCH, etc.)
16. Use `forecast_ts_utc` as the **index and deduplication key**.

---

## F. Athena Join Logic

17. Join condition:
    ```sql
    ON p.forecast_ts_utc = g.timestamp_utc
    ```
18. Preserve the extra forecast row:
    - Use `predictions LEFT JOIN gold`
    - Future row appears with `gold.* = NULL`.

---

## G. Power BI Visualization

19. Use `forecast_trade_date_et` for human-readable axes if desired.
20. Create a row label in Athena:
    ```sql
    CASE
      WHEN g.timestamp_utc IS NULL THEN 'forecast'
      ELSE 'historical'
    END AS row_type
    ```
21. Plot:
    - Realized series from gold columns
    - Forecast series from prediction columns
    - Last point is forecast-only.

---

## H. DST & Calendar Handling

22. Never hardcode UTC offsets.
    - Always convert between `America/New_York` and `UTC`.
23. Weekday-only calendars are acceptable initially.
24. For production, use a US trading calendar (NYSE/Nasdaq holidays).
25. Early closes:
    - Ideal: dynamic close time from calendar
    - Acceptable: assume 16:00 ET and document the assumption.

---

## I. Sanity Checks (highly recommended)

26. Assert gold timestamps are ET-close-UTC:
    - Daily equity data should be `20:00` or `21:00` UTC (DST dependent).
27. Assert forecast timestamps follow the same rule.
28. Prevent duplicate forecasts:
    - One row per `forecast_ts_utc`.

---

**Invariant:**  
If `gold.timestamp_utc` and `predictions.forecast_ts_utc` are identical by
construction, all joins, evaluations, and Power BI visuals will be correct.
