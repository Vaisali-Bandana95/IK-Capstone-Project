import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any

from src.config.logging import log_event
from src.db import run_query_safe

MAX_ID = 2_147_483_647
logger = logging.getLogger(__name__)


def log_tool_call(tool_name: str) -> None:
    log_event(logger, "tool.called", tool=tool_name)


def encode_json(value: Any) -> str:
    return json.dumps(value, default=str)


def validate_id(value: str, name: str) -> tuple[int | None, str | None]:
    # Refuse booleans, signs, decimals, excessive digit lengths and SQL fragments.
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,10}", value):
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


def best_fuzzy_match(phrase: str, values: list[str], threshold: float = 0.80) -> str | None:
    """Find a typo-tolerant match without adding fuzzy-lookup dependencies."""
    def normalized(text: str) -> str:
        return re.sub(r"[^\w]+", "", text.casefold())

    needle = normalized(phrase)
    if len(needle) < 3:
        return None
    ranked = sorted(
        ((SequenceMatcher(None, needle, normalized(value)).ratio(), value)
         for value in values if isinstance(value, str)),
        key=lambda item: (-item[0], item[1]),
    )
    return ranked[0][1] if ranked and ranked[0][0] >= threshold else None


def fuzzy_search_parameter(
    phrase: str,
    *,
    sql: str,
    name: str,
) -> tuple[str | None, str | None]:
    """Only use fixed, caller-owned SQL, never dynamic SQL from customer input."""
    rows, error = query_rows(sql, {})
    if error:
        return None, error
    candidate = best_fuzzy_match(phrase, [row[name] for row in rows])
    return candidate, None


def not_found(message: str) -> str:
    return encode_json({"message": message})
