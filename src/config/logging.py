import json
import logging
import os
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

_correlation_id: ContextVar[str | None] = ContextVar(
    "correlation_id",
    default=None,
)

_SAFE_FIELDS = {
    "duration_ms",
    "error_type",
    "outcome",
    "preference_count",
    "query_kind",
    "row_count",
    "route",
    "tool",
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=UTC,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "unstructured_log"),
        }
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            entry["correlation_id"] = correlation_id

        fields = getattr(record, "fields", {})
        if isinstance(fields, dict):
            entry.update(
                (key, value)
                for key, value in fields.items()
                if key in _SAFE_FIELDS
            )
        return json.dumps(entry, default=str)


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        raise ValueError(f"Invalid LOG_LEVEL: {level_name}")  # noqa: TRY004

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def set_correlation_id(correlation_id: str) -> Token[str | None]:
    return _correlation_id.set(correlation_id)


def reset_correlation_id(token: Token[str | None]) -> None:
    _correlation_id.reset(token)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    safe_fields = {
        key: value
        for key, value in fields.items()
        if key in _SAFE_FIELDS
    }
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "correlation_id": _correlation_id.get(),
            "fields": safe_fields,
        },
    )
