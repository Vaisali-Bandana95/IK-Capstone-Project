import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.config.logging import log_event

from .database import engine

logger = logging.getLogger(__name__)

_QUERY_KINDS = {"SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP"}


def _query_kind(sql: str) -> str:
    first_token = sql.lstrip().split(maxsplit=1)
    if not first_token:
        return "unknown"
    keyword = first_token[0].upper()
    return keyword.lower() if keyword in _QUERY_KINDS else "other"


def execute_query(
    sql: str,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    start_time = time.perf_counter()
    with engine.connect() as connection:
        rows = connection.execute(text(sql), dict(parameters or {})).mappings().all()

    log_event(
        logger,
        "database.query_completed",
        duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        query_kind=_query_kind(sql),
        row_count=len(rows),
        outcome="success",
    )
    return json.dumps([dict(row) for row in rows], default=str)


def run_query_safe(
    sql: str,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    try:
        return execute_query(sql, parameters)
    except SQLAlchemyError as error:
        log_event(
            logger,
            "database.query_failed",
            level=logging.ERROR,
            error_type=type(error).__name__,
            outcome="error",
        )
        return json.dumps(
            {"error": "Query execution failed", "code": "QUERY_ERROR"}
        )
