# Cloud Deployment Project Outline  
**Energy Volatility Forecasting Pipeline**

This document outlines the end-to-end design and deployment plan for an energy-market time series forecasting model on the cloud. The project covers data ingestion, model deployment, automation, and business intelligence visualization.

---

## Phase 0 — Preparation & Repository Readiness

**Objective:**  
Refactor the existing codebase so that data ingestion, training, and prediction can run as standalone, repeatable jobs in a cloud environment.

**Key Tasks:**
- Add a pipeline/orchestration layer:
  - `src/proj/pipelines/update_data.py`
  - `src/proj/pipelines/retrain.py`
  - `src/proj/pipelines/predict.py`
- Centralize configuration:
  - Local vs cloud storage (S3)
  - Retraining frequency
  - Model parameters
- Define stable schemas for:
  - Curated data
  - Features
  - Predictions

**Deliverables:**
- Pipelines runnable locally via CLI
- Clear separation between data access, transformations, and orchestration

---

## Phase 1 — Data Retrieval, Updating, Processing, and Cloud Storage

**Objective:**  
Automatically retrieve new data from multiple clean APIs, combine sources, and store processed outputs in cloud storage.

### 1.1 Data Storage Architecture
Use an S3-based data lake with layered structure:


s3://<bucket>/
├── raw/
│   └── source=<name>/
│       └── dt=YYYY-MM-DD/
│
├── curated/
│   └── dataset=<name>/
│       └── dt=YYYY-MM-DD/
│           └── data.parquet
│
├── features/
│   └── model=garchx/
│       └── dt=YYYY-MM-DD/
│           └── features.parquet
│
├── predictions/
│   └── model=garchx/
│       └── run_dt=YYYY-MM-DD/
│           └── preds.parquet
│
└── models/
    └── model=garchx/
        └── train_end=YYYY-MM-DD/
            ├── model.pkl
            └── metadata.json


### 1.2 Incremental Update Logic
- Determine the last available date from curated data or metadata
- Pull only new observations since that date from each API
- Merge multiple sources on a common time index (weekly)
- Run preprocessing, validation, and deduplication
- Persist  to S3 using date-based partitions

**Deliverables:**
- Incremental ingestion pipeline
- Reproducible, partitioned datasets in S3
- Basic data quality checks and logging

---

## Phase 2 — Dockerization and Model Inference

**Objective:**  
Package the environment, model code, and dependencies into a Docker image that can run consistently across environments.

### 2.1 Docker Setup
- Create a `Dockerfile` that:
  - Installs Python dependencies
  - Copies source code
  - Defines entrypoints for:
    - Data updates
    - Retraining
    - Prediction

### 2.2 Model Artifacts & Outputs
- Training job:
  - Fits model
  - Saves artifacts (`model.pkl`, metadata) to S3
- Prediction job:
  - Loads latest model
  - Reads newest features
  - Writes forecasts to S3

**Deliverables:**
- Docker image builds and runs locally
- Image pushed to cloud container registry (e.g., ECR)

---

## Phase 3 — Automation and Orchestration

**Objective:**  
Run the full pipeline automatically on a fixed schedule without manual intervention.

### 3.1 Scheduling
- Use a cloud scheduler (e.g., EventBridge) to trigger jobs weekly
- Typical execution order:
  1. Update data
  2. Build features
  3. Retrain model (weekly or monthly)
  4. Generate predictions

### 3.2 Orchestration Options
- **Simple:** Separate scheduled container tasks
- **Robust:** Step-based workflow orchestration with:
  - Retries
  - Failure handling
  - Notifications

### 3.3 Monitoring
- Centralized logs
- Alerts for job failures
- Metadata tracking (last successful run, model version)

**Deliverables:**
- Fully automated scheduled pipeline
- Monitoring and alerting enabled
- Idempotent, re-runnable jobs

---

## Phase 4 — Power BI Dashboard

**Objective:**  
Expose model outputs and historical performance through an interactive dashboard.

### 4.1 Data Access Options

**Option A: S3 + Athena**
- Store predictions and curated data as Parquet in S3
- Query via Athena
- Power BI connects to Athena and refreshes on schedule

**Option B: Serving Database**
- Keep S3 as the data lake
- Write latest predictions to a small relational database
- Power BI connects directly for low-latency queries

### 4.2 Dashboard Components
- Forecast overview (latest predictions)
- Historical performance and backtesting metrics
- Volatility drivers and explanatory variables
- Data freshness and pipeline health indicators

**Deliverables:**
- Power BI dataset and scheduled refresh
- Multi-page dashboard with clear timestamps and metadata

---

## Project Milestones

1. Local pipeline runs end-to-end
2. Data written to cloud storage
3. Dockerized pipeline runs locally
4. Scheduled cloud execution succeeds
5. Power BI dashboard connected and refreshing

---

## Estimated Timeline
- Phase 0–1: 1–2 days
- Phase 2: 0.5–1 day
- Phase 3: 1–2 days
- Phase 4: 1–2 days

---

## Summary
This project delivers a production-style, cloud-based time-series forecasting system with automated data ingestion, scheduled retraining and prediction, reproducible storage, and business-facing visualization.



