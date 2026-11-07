FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=America/Mexico_City

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata gcc libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/mcp/requirements-mcp.txt requirements-mcp.txt
RUN pip install --no-cache-dir -r requirements-mcp.txt

COPY src ./src
COPY scripts ./scripts
COPY mcp_server ./mcp_server
COPY sql ./sql

# Datos point-in-time que leen los agentes de dominio (mercado, noticias,
# fundamentales, features/ml, lab, evaluación) — ver src/agents_mcp/domains/.
COPY data/market ./data/market
COPY data/news ./data/news
COPY data/eval ./data/eval
COPY data/fundamentals.parquet ./data/fundamentals.parquet
COPY data/lab ./data/lab

EXPOSE 8080
CMD ["python", "-m", "mcp_server.server"]
