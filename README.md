# Energy Volatility Forecasting System  
**Author: Jack Yeung**


## Overview

An end-to-end, production-oriented system for **forecasting energy market volatility** using econometric models, macroeconomic indicators, financial market data, and weather-driven features.

This project emphasizes **time-series correctness**, **statistical rigor**, and **realistic deployment practices**, moving beyond simple regression-style prediction.


## Motivation

Energy markets exhibit volatility clustering, structural breaks, and sensitivity to macroeconomic and weather-related shocks. Accurate volatility forecasting is essential for:

- Risk management and hedging
- Portfolio construction and capital allocation
- Scenario analysis and stress testing

This project is designed to study and model these dynamics in a **statistically principled and scalable** way.


## Key Capabilities

- Econometric volatility models (EWMA, GARCH, GARCH-X, HAR-RV)
- Integration of macroeconomic, financial, and weather-based features
- Anomaly detection of weather features
- Walk-forward backtesting with proper volatility loss functions
- Config-driven, reproducible data pipelines
- Support for local and cloud-based execution


## Repository Structure

```

├── Dockerfile # Containerized execution
├── LICENSE
├── configs/ # Config-driven pipeline definitions
│ ├── run_all.yaml # Local pipeline configuration
│ ├── run_all_cloud.yaml # Cloud deployment configuration
│ └── steps/
│ ├── ingest.yaml # Data ingestion settings
│ ├── merge.yaml # Dataset alignment and joins
│ ├── build_features.yaml # Feature engineering
| ├── regime.yaml # markov regime 
│ └── prediction.yaml # Training and inference
│
├── data/
│ ├── bronze/ # Raw ingested data + ingestion state
│ ├── silver/ # Cleaned and standardized datasets
│ ├── gold/ # Model-ready tables
| ├── regimes/ #table of latent regime 
│ ├── prediction/ # Forecasts
| ├── aggregates/ # scored forecasts 
│ └── external/ # Static reference data
│
├── docs/
│ ├── Volatility_Modeling.pdf # Theoretical and modeling background
│ ├── architecture.md # System architecture overview
│ ├── cloud.md # Cloud deployment details
│ └── plan.md # Project planning and roadmap
│
├── figures/ # Diagnostics, forecasts, and visualizations
├── notebooks/ # Exploratory analysis and prototyping
│
├── src/
│ └── proj/
│ ├── data/ # Ingestion, storage, and merging logic
│ ├── features/ # Feature engineering and preprocessing
│ ├── models/ # Volatility models
│ ├── evaluation/ # Backtesting and scoring
│ ├── pipelines/ # Orchestration logic
│ ├── trading/ # Optional execution and brokerage hooks
│ └── utils/ # Config, paths, logging, and helpers
│
├── requirements.txt
├── requirements-cloud.txt
└── pyproject.toml
```



## Data Architecture

The project follows a **Bronze–Silver–Gold** data layering pattern to ensure reproducibility and transparency:

- **Bronze**  
  Raw data ingested from financial, macroeconomic, and weather APIs  
  Includes ingestion state files to support incremental updates

- **Silver**  
  Cleaned, validated, and frequency-aligned time-series datasets

- **Gold**  
  Feature-engineered, model-ready tables used for training and inference

This layered structure ensures **auditability**, **reproducibility**, and **clean separation of concerns** across the pipeline.



## Models Implemented

The system focuses explicitly on **volatility modeling**, rather than return prediction.

Implemented models include:

- **EWMA** (Exponentially Weighted Moving Average)
- **GARCH**
- **GARCH-X** (GARCH with exogenous variables)
- **HAR-RV**
- **HAR-RV-X** (HAR-RV with exogenous variables)
- **Tree-based volatility models (XGBoost)**

All models conform to a shared interface, enabling consistent training, forecasting, and evaluation across model classes.



## Feature Engineering

Feature construction is guided by economic intuition and empirical finance practice, including:

- Lagged returns and absolute returns
- Realized variance and log-variance transformations
- Macroeconomic indicators (interest rates, spreads, volatility indices)
- Weather-based demand proxies and anomaly features
- Automated preprocessing and transformation pipelines

All feature definitions are **configurable, reproducible, and extensible**.



## Evaluation Methodology

Model performance is assessed using **time-series–aware backtesting**:

- Rolling or expanding training windows
- No random splits or look-ahead bias
- Forecasts evaluated against realized variance

Primary evaluation metric:

- **QLIKE loss**, a proper scoring rule for volatility forecasts

Secondary diagnostics include RMSE, forecast stability checks, and visual inspection of volatility dynamics.

> **Full Write-Up:**  
> A detailed discussion of the modeling framework, assumptions, and empirical results is provided in  
> `docs/Volatility_Modeling.pdf`.



## Running the Pipeline

### Local Execution
From the repository root, first set up the environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
Then run the pipeline:
```bash
python src/proj/pipelines/run_all.py --cfg configs/run_all.yaml

