FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config.yaml ./config.yaml

# Состояние и логи держим на томах, чтобы переживали пересборку контейнера.
RUN mkdir -p /app/data /app/logs
VOLUME ["/app/data", "/app/logs"]

# Бот — долгоживущий процесс с long polling; healthcheck смотрит, что БД пишется.
HEALTHCHECK --interval=5m --timeout=10s --start-period=1m --retries=3 \
    CMD python -c "import pathlib,sys; sys.exit(0 if pathlib.Path('/app/data/state.db').exists() else 1)"

CMD ["python", "-m", "app.main"]
