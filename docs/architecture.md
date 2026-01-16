# Cloud Architecture Overview

This document describes the end-to-end cloud architecture for the Energy Volatility Forecasting project.
The system supports incremental ingestion, feature building, automatic model fitting and retraining,
daily (business-day) forecasting, and BI reporting through an S3 → Glue → Athena → Power BI stack.

---

## 1. Design Principles

1. Batch-first, scheduled execution  
   Forecasts are generated on a defined cadence (business-day / daily). No real-time API is required.

2. S3 as the system of record  
   All curated datasets, engineered features, predictions, and evaluation outputs are stored in S3.

3. Auto-fit and reproducibility  
   Models are fit automatically from config and data, but every run is traceable via run metadata,
   partitions, and artifacts.

4. Separation of concerns  
   Ingestion, feature engineering, model training, inference, and scoring are modular pipeline stages.

5. Query-first analytics  
   Downstream reporting is driven by Athena SQL over S3, with schema managed by AWS Glue.

---

## 2. High-Level Architecture

### Major Layers
1. Data Sources  
2. Containerized Pipeline (Batch Compute)  
3. S3 Data Lake + Artifacts  
4. Glue Catalog + Athena Query Layer  
5. Power BI Dashboards  

### High-Level Flow

External APIs / Data Feeds  
↓  
Scheduled Container Job (ECS Fargate via EventBridge)  
- Incremental ingest  
- Feature engineering  
- Auto-fit / retrain (configurable)  
- Forecast + scoring  
↓  
S3 Data Lake (Parquet, partitioned)  
↓  
Glue Crawler / Catalog Updates  
↓  
Athena Views / Queries  
↓  
Power BI (ODBC / Athena connector)

---

## 3. Data Sources

The pipeline integrates multiple domains:

- Market data (prices, returns, realized variance / realized volatility)
- Macroeconomic series (e.g., FRED: rates, spreads, volatility indices)
- Weather and climate signals (anomalies, shocks, derived indicators)
- Static reference datasets (optional)

All sources are normalized to a shared time index, typically daily business-day.

---

## 4. Ingestion Layer

### Purpose
Incrementally fetch new data since the last successful run, standardize schemas,
and write append-only partitions.

### Responsibilities
- Track last ingested timestamp per source
- Pull only missing increments (idempotent re-runs)
- Standardize timezones, indices, and column naming
- Validate continuity and missingness rules
- Write outputs to S3 as partitioned Parquet

### Outputs
- Curated modeling inputs stored in S3 (gold layer)

---

## 5. Feature Engineering Layer

### Purpose
Create model-ready feature matrices and derived signals consistently across runs.

### Responsibilities
- Build lags, rolling windows, and transforms
- Construct cross-domain features (market, macro, weather)
- Enforce look-ahead safety
- Align targets with as-of timestamps
- Materialize feature sets by experiment configuration

### Outputs
- Feature datasets partitioned by run_id and as-of date

---

## 6. Modeling Layer (Auto-Fit / Auto-Train)

### Purpose
Automatically fit volatility models based on configuration and available data.

### Supported Models
- EWMA baselines
- GARCH and GARCH-X variants
- HAR-RV style regressions
- ML benchmarks (e.g., gradient-boosted models)
- Future extensions: regime-switching models

### Training Strategy
- Scheduled or conditional retraining
- Rolling or expanding training windows
- Model-specific estimation methods (e.g., MLE)

### Outputs
- Serialized model artifacts in S3
- Training metadata and diagnostics
- Versioned by run_id and training cutoff date

---

## 7. Prediction and Scoring Layer

### Purpose
Generate forecasts and evaluation metrics in the same pipeline run.

### Responsibilities
- Load latest model artifacts
- Produce forward forecasts for the configured horizon
- Join predictions with realized outcomes when available
- Compute metrics such as QLIKE and RMSE
- Generate regime labels or percentiles if configured

### Outputs
- predictions/ : point forecasts and metadata
- aggregates/  : scored results and rolling performance summaries

---

## 8. Storage Layer (S3 Data Lake)

### Primary Storage
- Amazon S3

### Logical Layout
- bronze/       raw ingested data (optional)
- silver/       standardized tables (optional)
- gold/         curated modeling dataset
- features/     model-ready feature matrices
- models/       serialized model artifacts
- predictions/  forecast outputs
- aggregates/   evaluation metrics and rollups
- athena_results/ query outputs

S3 is treated as the single source of truth.

---

## 9. Orchestration and Automation

### Scheduling
- EventBridge triggers the pipeline on a business-day cadence.

### Execution
- A single ECS Fargate container runs:
  1. Ingestion
  2. Feature engineering
  3. Model fitting or retraining
  4. Prediction
  5. Scoring and aggregation
  6. State update and persistence

### Observability
- Centralized logging
- Fail-fast execution
- Run-level traceability via run_id

---

## 10. Glue and Athena Query Layer

### Purpose
Expose S3 datasets for analytics and BI consumption.

### Components
- AWS Glue Data Catalog for schema and partitions
- Glue Crawlers or managed table definitions
- Athena SQL for querying and view creation

### BI-Oriented Outputs
- dashboard_fact view (predictions + realized + regimes)
- model_performance view (rolling metrics by model)
- data_freshness view (last successful run and coverage)

---

## 11. Visualization Layer (Power BI)

### Data Access
Power BI connects to Athena using the ODBC or native connector.

### Dashboard Focus
- Latest volatility forecasts
- Historical forecast vs realized comparison
- Model-level performance trends
- Regime and percentile indicators
- Data freshness and operational status

The dashboard is read-only and does not trigger compute.

---

## 12. Experimentation and Extensibility

- New models can be added via configuration
- Feature sets are experiment-specific
- Multiple runs and model variants coexist via partitioning
- Historical forecasts remain fully reproducible

Planned extensions include regime-switching models,
forecast ensembles, and enhanced monitoring.

---

## 13. Summary

This architecture provides a production-style batch system for volatility forecasting:

- Compute: ECS Fargate (scheduled batch jobs)
- Storage: S3 data lake (partitioned Parquet)
- Query: Glue Data Catalog + Athena
- Visualization: Power BI over Athena views

It balances research flexibility with operational robustness.
