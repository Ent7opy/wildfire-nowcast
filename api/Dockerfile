FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app

WORKDIR $APP_HOME

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgdal36 \
        gdal-bin \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN python -m pip install --upgrade pip uv \
    && uv sync --frozen --no-dev

COPY . ./api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

