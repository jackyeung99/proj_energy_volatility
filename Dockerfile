# syntax=docker/dockerfile:1
# ---- Base image ----
FROM python:3.10-slim

# ---- Runtime env ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # make `src/` importable (your package lives in src/proj)
    PYTHONPATH=/app/src

WORKDIR /app

# ---- System deps (common for pandas/pyarrow/arch/xgboost etc.) ----
# If you don't use some of these, you can remove later.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    # often needed by numpy/pandas wheels; also useful for xgboost/arch edge cases
    libgomp1 \
    # SSL/HTTP
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python deps first (better caching) ----
# If you rely on pyproject for packaging, you can switch to `pip install -e .`
# COPY requirements.txt /app/requirements.txt
# RUN python -m pip install --upgrade pip && \
#     pip install -r /app/requirements.txt

# Optional: if you prefer installing from pyproject.toml instead of requirements.txt:
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install -e .

# ---- Copy project files ----
# Keep only what you need at runtime (configs + src). Data can be mounted or read from S3 in cloud.
COPY src /app/src
COPY configs /app/configs

# Optional: if you want the container to ship with your existing local parquet files
# (usually NOT recommended for ECS; better to pull from S3)
# COPY data /app/data

# ---- Create a writable data dir (useful when mounting a volume) ----
RUN mkdir -p /app/data

# ---- Default command ----
# Uses your existing entrypoint: src/proj/pipelines/run_all.py
# You can override the config at runtime:
#   docker run ... -e CFG_PATH=/app/configs/run_all_cloud.yaml ...
ENV CFG_PATH=/app/configs/run_all_cloud.yaml

CMD ["sh", "-lc", "python -m proj.pipelines.run_all --cfg ${CFG_PATH}"]
