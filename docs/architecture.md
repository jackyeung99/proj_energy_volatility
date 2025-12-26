# Cloud Architecture Overview

This document describes the end-to-end architecture for the Energy Volatility Modeling project.  
The system is designed to support **incremental data ingestion**, **batch time-series modeling**, **automated retraining and forecasting**, and **business intelligence visualization**, while remaining flexible for experimentation and scalable to cloud deployment.

---

## 1. Design Principles

The architecture is guided by the following principles:

1. **Batch-first, not API-first**  
   Forecasts are generated on a scheduled basis and written to storage. No real-time inference is required.

2. **S3 as the system of record**  
   All curated data, features, model artifacts, and predictions are stored in object storage.

3. **Separation of concerns**  
   Data ingestion, feature engineering, modeling, and evaluation are implemented as independent pipeline stages.

4. **Config-driven experimentation**  
   Model behavior is controlled through configuration files rather than code changes.

5. **Reproducibility over optimization**  
   Each run can be traced to specific data partitions, model parameters, and code versions.

---

## 2. High-Level Architecture

The system consists of four major layers:

1. **Data Sources**
2. **Data & Modeling Pipelines**
3. **Storage Layer**
4. **Analytics & Visualization**

### High-Level Flow

External APIs
↓
Data Ingestion Pipeline
↓
Curated Weekly Dataset (S3)
↓
Feature Engineering Pipeline
↓
Model Training & Prediction
↓
Predictions Dataset (S3)
↓
Power BI Dashboard


---

## 3. Data Sources

The project integrates multiple external data sources, including:

- Financial market data (energy equities, returns, volatility)
- Macroeconomic indicators (e.g., FRED series)
- Weather and climate variables (temperature, precipitation, wind shocks)
- Static reference datasets (e.g., plant or energy infrastructure data)

All sources are normalized to a **weekly frequency** with a shared time index (`week_end`).

---

## 4. Data Ingestion Layer

### Purpose
The ingestion layer retrieves new data incrementally, processes it into a canonical schema, validates it, and persists it to cloud storage.

### Responsibilities
- Determine the last successfully ingested date
- Pull only new observations since that date
- Clean and standardize source-specific formats
- Merge data from multiple sources
- Validate time-series integrity
- Write curated datasets to storage

### Output
- A single **curated weekly dataset** containing all inputs required for modeling

### Key Properties
- Incremental (supports weekly updates)
- Idempotent (safe to re-run)
- Source-agnostic (each data source is isolated)

---

## 5. Feature Engineering Layer

### Purpose
Transform curated data into model-ready features while remaining flexible for experimentation.

### Responsibilities
- Construct lagged variables
- Apply transformations and scaling
- Select feature subsets based on experiment configuration
- Drop invalid observations created by lagging

### Output
- A **features dataset** that matches the exact input expected by the model

### Design Choice
Feature construction is **experiment-specific**.  
The curated dataset is never modified to suit a particular model.

---

## 6. Modeling Layer

### Purpose
Estimate time-series models and generate forecasts.

### Supported Models
- ARIMA
- GARCH
- GARCH-X
- Gradient-boosted models (XGBoost) for comparison

### Training Strategy
- Batch retraining on a fixed schedule (e.g., weekly)
- Expanding or rolling windows, configurable
- Parameters are re-estimated via maximum likelihood (not online learning)

### Outputs
- Serialized model artifacts
- Model metadata (training window, feature set, parameters)

Artifacts are written to cloud storage and versioned by training date.

---

## 7. Prediction Layer

### Purpose
Generate forward-looking forecasts using the most recent trained model.

### Responsibilities
- Load latest model artifacts
- Read most recent feature data
- Produce forecasts for a specified horizon
- Write predictions with timestamps and identifiers

### Output
- A **predictions dataset** containing forecasts and metadata

Predictions are partitioned by run date to support historical analysis and backtesting.

---

## 8. Storage Layer

### Primary Storage
- **Amazon S3**

### Logical Layout
- `curated/` — cleaned, merged weekly inputs
- `features/` — model-ready feature matrices
- `models/` — trained model artifacts
- `predictions/` — forecast outputs

### Rationale
- Cheap and durable storage
- Natural fit for partitioned time-series data
- Directly queryable via analytics tools (Athena, Power BI)

S3 is treated as the **single source of truth**.

---

## 9. Orchestration & Automation

### Scheduling
- A weekly schedule triggers the entire pipeline.

### Execution
- A single containerized batch job runs:
  1. Data ingestion
  2. Feature engineering
  3. Model training (optional per run)
  4. Prediction

### Logging & Monitoring
- All pipeline steps emit logs
- Failures halt execution and prevent state updates

This design minimizes operational complexity while remaining extensible.

---

## 10. Visualization Layer (Power BI)

### Data Access
Power BI reads forecast and historical data from:
- Cloud storage via a query layer (e.g., Athena), or
- A lightweight serving database populated from S3 (optional)

### Dashboard Focus
- Forecasted volatility and returns
- Model performance metrics
- Temporal comparisons and regime changes
- Data freshness indicators

The dashboard is **read-only** and does not interact with the pipeline.

---

## 11. Experimentation & Extensibility

The architecture supports experimentation by design:

- New models can be added without changing ingestion logic
- Feature sets are selected via configuration
- Multiple experiments can coexist using versioned outputs
- Historical runs remain reproducible

Future extensions may include:
- Additional data sources
- Model ensembles
- Regime-switching models
- Real-time inference endpoints (if needed)

---

## 12. Summary

This architecture provides a clear, modular, and scalable framework for deploying time-series volatility models in the cloud. It balances engineering discipline with research flexibility and is well-suited for both academic exploration and production-style workflows.
