import json
from typing import Any, Mapping

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .database import engine


def execute_query(
    sql: str,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    with engine.connect() as connection:
        rows = connection.execute(text(sql), dict(parameters or {})).mappings().all()

    return json.dumps([dict(row) for row in rows], default=str)


def run_query_safe(
    sql: str,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    try:
        return execute_query(sql, parameters)
    except SQLAlchemyError:
        return json.dumps(
            {"error": "Query execution failed", "code": "QUERY_ERROR"}
        )
