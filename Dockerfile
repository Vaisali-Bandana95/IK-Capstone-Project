FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MEMORY_DB_PATH=/data/preferences.sqlite3 \
    CHECKPOINT_DB_PATH=/data/checkpoints.sqlite3 \
    CHINOOK_CACHE_PATH=/data/Chinook_Sqlite.sql \
    CHINOOK_SQL_URL=https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /data \
    && chown -R app:app /app /data

COPY app.py .
COPY src/ ./src/
RUN chown -R app:app /app

EXPOSE 7860
VOLUME ["/data"]

USER app
CMD ["python", "app.py"]
