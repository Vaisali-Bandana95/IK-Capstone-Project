import json
import re
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from src.db import run_query_safe
from src.graph import (
    UserInput,
    build_verification_graph,
    create_verification_node,
    route_after_verification,
)


class EchoIdentifierLLM:
    def with_structured_output(self, schema: type[UserInput]) -> "EchoIdentifierLLM":
        assert schema is UserInput
        return self

    def invoke(self, messages: list[Any]) -> UserInput:
        return UserInput(identifier=str(messages[-1].content))


class NumericIdentifierLLM(EchoIdentifierLLM):
    def invoke(self, messages: list[Any]) -> UserInput:
        match = re.search(r"\b\d+\b", str(messages[-1].content))
        return UserInput(identifier=match.group() if match else "")


class CredentialIdentifierLLM(EchoIdentifierLLM):
    def invoke(self, messages: list[Any]) -> UserInput:
        message = str(messages[-1].content)
        match = re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+|\b\d+\b", message)
        return UserInput(identifier=match.group() if match else "")


def get_customer_contact(customer_id: int) -> dict[str, str]:
    import json

    rows = json.loads(
        run_query_safe(
            """
            SELECT Email AS email, Phone AS phone
            FROM Customer WHERE CustomerId = :customer_id
            """,
            {"customer_id": customer_id},
        )
    )
    return rows[0]


def test_identifier_schema_has_single_identifier_field() -> None:
    identifier = UserInput(identifier="1")

    assert identifier.identifier == "1"
    assert set(UserInput.model_fields) == {"identifier"}


@pytest.mark.parametrize("identifier", ["1", " 1 "])
def test_verification_finds_customer_by_numeric_id(identifier: str) -> None:
    node = create_verification_node(EchoIdentifierLLM())

    result = node({"messages": [HumanMessage(content=identifier)]})

    assert result["customer_id"] == 1
    assert result["verified"] is True
    assert any(isinstance(message, SystemMessage) for message in result["messages"])
    assert "customer_id is 1" in result["messages"][0].content


def test_verification_matches_email_case_insensitively() -> None:
    email = get_customer_contact(1)["email"]
    node = create_verification_node(EchoIdentifierLLM())

    result = node({"messages": [HumanMessage(content=email.swapcase())]})

    assert result["customer_id"] == 1


def test_verification_matches_normalized_phone() -> None:
    phone = get_customer_contact(1)["phone"]
    assert phone
    node = create_verification_node(EchoIdentifierLLM())

    result = node({"messages": [HumanMessage(content=phone.replace("-", " "))]})

    assert result["customer_id"] == 1


def test_employee_phone_cannot_verify_a_customer_account() -> None:
    employee_phone = json.loads(
        run_query_safe(
            "SELECT Phone FROM Employee WHERE EmployeeId = :employee_id",
            {"employee_id": 8},
        )
    )[0]["Phone"]
    assert employee_phone == "+1 (403) 467-3351"
    node = create_verification_node(EchoIdentifierLLM())

    result = node(
        {"messages": [HumanMessage(content=f"My phone number is {employee_phone}")]}
    )

    assert result["verified"] is False
    assert "customer_id" not in result
    assert "couldn't find an account" in result["verification_error"]


def test_verification_extracts_labeled_phone_without_llm_extraction() -> None:
    class UnexpectedExtractor(EchoIdentifierLLM):
        def invoke(self, messages: list[Any]) -> UserInput:
            raise AssertionError("Labeled phone should be extracted deterministically")

    phone = get_customer_contact(1)["phone"]
    node = create_verification_node(UnexpectedExtractor())

    result = node(
        {"messages": [HumanMessage(content=f"My phone number is {phone}")]}
    )

    assert result["verified"] is True
    assert result["customer_id"] == 1


def test_verification_matches_phone_using_last_ten_digits() -> None:
    phone = get_customer_contact(1)["phone"]
    local_number = "".join(character for character in phone if character.isdigit())[-10:]
    node = create_verification_node(EchoIdentifierLLM())

    result = node(
        {"messages": [HumanMessage(content=f"My phone number is {local_number}")]}
    )

    assert result["verified"] is True
    assert result["customer_id"] == 1


def test_verification_matches_digits_only_phone_when_not_a_customer_id() -> None:
    phone = get_customer_contact(1)["phone"]
    digits = "".join(character for character in phone if character.isdigit())
    if digits == "1":
        pytest.skip("Phone digits unexpectedly equal the customer ID")

    node = create_verification_node(EchoIdentifierLLM())
    result = node({"messages": [HumanMessage(content=digits)]})

    assert result["customer_id"] == 1


def test_invalid_identifier_does_not_verify_customer() -> None:
    node = create_verification_node(EchoIdentifierLLM())

    result = node({"messages": [HumanMessage(content="not a customer credential")]})

    assert result["verified"] is False
    assert "customer_id" not in result


def test_unknown_customer_id_reports_not_found_and_requests_another_identifier() -> None:
    node = create_verification_node(NumericIdentifierLLM())

    result = node({"messages": [HumanMessage(content="My ID is 99999")]})

    assert result["verified"] is False
    assert result["verification_error"] == (
        "I couldn't find an account matching that identifier. Please try a "
        "different Customer ID, email address, or phone number."
    )
    assert "customer_id" not in result


def test_explicit_unknown_id_is_checked_as_written_not_llm_reinterpretation() -> None:
    class MisextractingLLM(EchoIdentifierLLM):
        def invoke(self, messages: list[Any]) -> UserInput:
            return UserInput(identifier="12")

    node = create_verification_node(MisextractingLLM())

    result = node({"messages": [HumanMessage(content="My ID is 99999")]})

    assert result["verified"] is False
    assert "couldn't find an account" in result["verification_error"]
    assert "customer_id" not in result


def test_unknown_customer_id_interrupt_can_resume_with_valid_email() -> None:
    graph = build_verification_graph(llm=CredentialIdentifierLLM())
    config = {"configurable": {"thread_id": "unknown-customer-id-test"}}

    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="My ID is 99999")]},
        config=config,
    )

    assert interrupted["verified"] is False
    snapshot = graph.get_state(config)
    prompt = snapshot.tasks[0].interrupts[0].value
    assert "couldn't find an account" in prompt
    assert "try a different Customer ID" in prompt

    email = get_customer_contact(1)["email"]
    resumed = graph.invoke(Command(resume=email), config=config)

    assert resumed["verified"] is True
    assert resumed["customer_id"] == 1


def test_failed_new_identifier_does_not_reuse_existing_verified_customer() -> None:
    node = create_verification_node(NumericIdentifierLLM())

    result = node(
        {
            "messages": [HumanMessage(content="My ID is 99999")],
            "customer_id": 1,
            "verified": True,
        }
    )

    assert result["verified"] is False
    assert "couldn't find an account" in result["verification_error"]


def test_verification_does_not_switch_accounts_in_an_existing_thread() -> None:
    node = create_verification_node(NumericIdentifierLLM())

    result = node(
        {
            "messages": [HumanMessage(content="My ID is 2")],
            "customer_id": 1,
            "verified": True,
        }
    )

    assert result["verified"] is False
    assert "already associated with a different account" in result[
        "verification_error"
    ]
    assert "customer_id" not in result


def test_verification_rejects_identifier_not_present_in_message() -> None:
    class HallucinatingLLM(EchoIdentifierLLM):
        def invoke(self, messages: list[Any]) -> UserInput:
            return UserInput(identifier="1")

    node = create_verification_node(HallucinatingLLM())

    result = node({"messages": [HumanMessage(content="Please verify my account")]})

    assert result["verified"] is False
    assert "customer_id" not in result


def test_verification_rejects_multiple_conflicting_identifiers() -> None:
    email = get_customer_contact(1)["email"]
    node = create_verification_node(NumericIdentifierLLM())

    result = node(
        {"messages": [HumanMessage(content=f"My ID is 2 and my email is {email}")]}
    )

    assert result["verified"] is False
    assert "multiple identifiers" in result["verification_error"]
    assert "customer_id" not in result


def test_conditional_edge_routes_only_verified_customer_forward() -> None:
    assert route_after_verification({"customer_id": 1, "verified": True}) == "next"
    assert route_after_verification({"verified": False}) == "human_input"
    assert route_after_verification({"customer_id": 1, "verified": False}) == "human_input"
    assert route_after_verification({"retry_count": 5}) == "exhausted"


def test_graph_interrupts_then_resumes_verification() -> None:
    graph = build_verification_graph(llm=EchoIdentifierLLM())
    config = {"configurable": {"thread_id": "verification-resume-test"}}

    interrupted = graph.invoke(
        {"messages": [HumanMessage(content="not a valid credential")]},
        config=config,
    )

    assert interrupted["verified"] is False
    snapshot = graph.get_state(config)
    assert snapshot.tasks
    assert snapshot.tasks[0].interrupts
    assert "Customer ID, email address, or phone number" in (
        snapshot.tasks[0].interrupts[0].value
    )

    resumed = graph.invoke(Command(resume="1"), config=config)

    assert resumed["customer_id"] == 1
    assert resumed["verified"] is True
    assert any(
        isinstance(message, SystemMessage)
        and "customer_id is 1" in message.content
        for message in resumed["messages"]
    )


def test_graph_uses_verified_customer_id_not_other_message_content() -> None:
    graph = build_verification_graph(llm=EchoIdentifierLLM())
    config = {"configurable": {"thread_id": "verification-state-test"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="1")]},
        config=config,
    )

    assert result["customer_id"] == 1
    injected = [
        message
        for message in result["messages"]
        if isinstance(message, SystemMessage)
    ]
    assert len(injected) == 1


def test_graph_stops_after_five_failed_verification_attempts() -> None:
    graph = build_verification_graph(llm=EchoIdentifierLLM())
    config = {"configurable": {"thread_id": "verification-retry-limit-test"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="not a credential")]},
        config=config,
    )
    for attempt in range(4):
        result = graph.invoke(Command(resume=f"invalid-{attempt}"), config=config)

    assert result["verified"] is False
    assert result["retry_count"] == 5
    assert "Maximum verification attempts" in result["verification_error"]
