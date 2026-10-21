import json

import pytest
from sqlalchemy import create_engine

from src.db import (
    execute_query,
    health_check,
    normalize_phone,
    run_query_safe,
    verify_database,
)
from src.db import database as database_module
from src.config.settings import load_settings


def test_execute_query_basic_select() -> None:
    assert json.loads(execute_query("SELECT 1 AS value")) == [{"value": 1}]


def test_execute_query_binds_parameters() -> None:
    result = execute_query(
        "SELECT :name AS name, :count AS count",
        {"name": "Chinook", "count": 3},
    )

    assert json.loads(result) == [{"name": "Chinook", "count": 3}]


def test_execute_query_returns_empty_list_for_no_rows() -> None:
    result = execute_query(
        "SELECT name FROM Artist WHERE ArtistId = :artist_id",
        {"artist_id": -1},
    )

    assert json.loads(result) == []


def test_execute_query_output_is_valid_json() -> None:
    result = execute_query("SELECT 'valid' AS value")

    assert json.loads(result) == [{"value": "valid"}]


def test_run_query_safe_returns_customer_count_as_json() -> None:
    result = run_query_safe("SELECT COUNT(*) FROM Customer")

    assert json.loads(result) == [{"COUNT(*)": 59}]


def test_run_query_safe_binds_parameters() -> None:
    result = run_query_safe(
        "SELECT CustomerId FROM Customer WHERE CustomerId = :cid",
        {"cid": 1},
    )

    assert json.loads(result) == [{"CustomerId": 1}]


def test_run_query_safe_returns_empty_list() -> None:
    result = run_query_safe(
        "SELECT CustomerId FROM Customer WHERE CustomerId = :cid",
        {"cid": -1},
    )

    assert json.loads(result) == []


def test_run_query_safe_returns_json_error_for_invalid_query() -> None:
    result = run_query_safe("SELECT * FROM MissingTable")

    assert json.loads(result) == {
        "error": "Query execution failed",
        "code": "QUERY_ERROR",
    }


@pytest.mark.parametrize(
    ("phone", "expected"),
    [
        ("+1 (555) 123-4567", "+15551234567"),
        ("+1 (212) 555-0198", "+12125550198"),
        ("2125550198", "2125550198"),
        ("212-555-0198", "2125550198"),
        ("", ""),
        (None, ""),
        ("++44 20 1234 5678", "+442012345678"),
    ],
)
def test_normalize_phone(phone: str | None, expected: str) -> None:
    assert normalize_phone(phone) == expected


def test_health_check_passes_for_initialized_chinook_database() -> None:
    assert health_check() is True


def test_verify_database_returns_healthy_status_and_table_count() -> None:
    assert verify_database() == {"healthy": True, "table_count": 11}


def test_health_check_fails_when_required_tables_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_engine = create_engine("sqlite://")
    monkeypatch.setattr(database_module, "engine", empty_engine)

    try:
        assert health_check() is False
        assert verify_database() == {"healthy": False, "table_count": 0}
    finally:
        empty_engine.dispose()


def test_settings_load_values_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.example.test")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "9000")

    settings = load_settings()

    assert settings.api_key == "test-key"
    assert settings.model_name == "test-model"
    assert settings.temperature == 0.4
    assert settings.api_base_url == "https://api.example.test"
    assert settings.port == 9000


def test_settings_accept_model_name_environment_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("MODEL_NAME", "alias-model")

    assert load_settings().model_name == "alias-model"


@pytest.mark.parametrize(
    ("variable", "value"),
    [("LLM_TEMPERATURE", "3"), ("GRADIO_SERVER_PORT", "70000")],
)
def test_settings_reject_out_of_range_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError):
        load_settings()
