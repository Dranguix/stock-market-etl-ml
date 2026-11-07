FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=America/Mexico_City

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/cloudrun/requirements-job.txt requirements-job.txt
RUN pip install --no-cache-dir -r requirements-job.txt

COPY src ./src
COPY scripts ./scripts
COPY sql ./sql
COPY data/signals ./data/signals
# Factores point-in-time del satélite (src/agent/core_satellite.py).
COPY data/fundamentals.parquet ./data/fundamentals.parquet
COPY data/eps ./data/eps
COPY data/dividends ./data/dividends

CMD ["python", "scripts/gcp_daily_job.py", "--mode", "ingest"]
