import json
import re
from typing import Any

from src.db import run_query_safe

MAX_ID = 2_147_483_647


def encode_json(value: Any) -> str:
    return json.dumps(value, default=str)


def validate_id(value: str, name: str) -> tuple[int | None, str | None]:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        return None, encode_json(
            {"error": f"{name} must be a positive integer", "code": "INVALID_INPUT"}
        )

    numeric_value = int(value)
    if not 1 <= numeric_value <= MAX_ID:
        return None, encode_json(
            {"error": f"{name} must be a positive integer", "code": "INVALID_INPUT"}
        )

    return numeric_value, None


def query_rows(
    sql: str,
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    result = json.loads(run_query_safe(sql, parameters))
    if isinstance(result, dict) and "error" in result:
        return None, encode_json(result)
    return result, None


def escaped_like(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def not_found(message: str) -> str:
    return encode_json({"message": message})
