FROM python:3.9-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PORT=8000 \
    STREAMLIT_PORT=8501 \
    MODEL_PATH=/app/models/hallucination_classifier

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

RUN useradd -m -u 1000 appuser

RUN mkdir -p /app/models /app/data /app/logs \
    && chown -R appuser:appuser /app

RUN python -m pytest tests/ -v

USER appuser

EXPOSE 8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

    ENV MODULE_NAME=hallucination_hunter.api.main \
    VARIABLE_NAME=app \
    API_PORT=8000 \
    UI_PORT=8501 \
    WORKERS=4 \
    LOG_LEVEL=info \
    HOST=0.0.0.0

VOLUME ["/app/models", "/app/data", "/app/logs"]

COPY --chown=appuser:appuser scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]


CMD ["sh", "-c", "uvicorn ${MODULE_NAME}:${VARIABLE_NAME} --host ${HOST} --port ${API_PORT} --workers ${WORKERS} --log-level ${LOG_LEVEL} & streamlit run hallucination_hunter/ui/app.py --server.port=${UI_PORT} --server.address=${HOST}"]
