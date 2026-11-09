import json
import logging
from io import StringIO

from src.config.logging import (
    JsonLogFormatter,
    log_event,
    reset_correlation_id,
    set_correlation_id,
)


def test_json_logging_includes_correlation_and_safe_fields_only() -> None:
    output = StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonLogFormatter())
    logger = logging.getLogger("test.safe_json_logging")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    token = set_correlation_id("request-123")
    try:
        log_event(
            logger,
            "tool.called",
            tool="get_track_details",
            email="customer@example.com",
            phone="+1 555 123 4567",
            user_message="show my invoices",
            invoice_contents="private",
        )
    finally:
        reset_correlation_id(token)
        logger.handlers.clear()
        logger.propagate = True

    record = json.loads(output.getvalue())
    assert record["event"] == "tool.called"
    assert record["correlation_id"] == "request-123"
    assert record["tool"] == "get_track_details"
    assert "customer@example.com" not in output.getvalue()
    assert "+1 555 123 4567" not in output.getvalue()
    assert "show my invoices" not in output.getvalue()
    assert "private" not in output.getvalue()


def test_log_event_does_not_retain_correlation_after_reset() -> None:
    output = StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonLogFormatter())
    logger = logging.getLogger("test.correlation_reset")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    token = set_correlation_id("request-456")
    reset_correlation_id(token)
    try:
        log_event(logger, "chat.request_started")
    finally:
        logger.handlers.clear()
        logger.propagate = True

    record = json.loads(output.getvalue())
    assert "correlation_id" not in record


def test_formatter_does_not_emit_unstructured_log_messages() -> None:
    output = StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonLogFormatter())
    logger = logging.getLogger("test.unstructured_message")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        logger.info("Customer email customer@example.com purchased tracks")
    finally:
        logger.handlers.clear()
        logger.propagate = True

    record = json.loads(output.getvalue())
    assert record["event"] == "unstructured_log"
    assert "customer@example.com" not in output.getvalue()
    assert "purchased tracks" not in output.getvalue()
