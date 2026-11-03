import json
import re
from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Checkpointer, interrupt
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from src.config import settings
from src.db import normalize_phone, run_query_safe

IDENTIFIER_EXTRACTION_PROMPT = """Extract exactly one customer-provided identifier from the message.
The identifier must be a numeric customer ID, an email address, or a phone number.
Return only the identifier as it appears in the message. Do not infer or invent an
identifier. If the message contains no such identifier, return an empty identifier."""

VERIFICATION_FALLBACK_PROMPT = (
    "I couldn't verify your account. Please try again with your Customer ID, "
    "email address, or phone number."
)


class UserInput(BaseModel):
    identifier: str = Field(
        description=(
            "Exactly one customer-provided numeric Customer ID, email address, "
            "or phone number, copied from the message. Empty if none is present."
        )
    )


class VerificationState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: int
    verified: bool
    verification_error: str
    retry_count: int


def _make_extractor(llm: Any | None) -> Runnable:
    if llm is None:
        if not settings.api_key:
            raise ValueError("OPENAI_API_KEY must be configured for verification")
        if not settings.model_name:
            raise ValueError("OPENAI_MODEL must be configured for verification")

        llm = ChatOpenAI(
            api_key=settings.api_key,
            model=settings.model_name,
            temperature=settings.temperature,
            base_url=settings.api_base_url,
        )
    return llm.with_structured_output(UserInput)


def _latest_human_message(state: VerificationState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _classify_identifier(identifier: str) -> Literal["customer_id", "email", "phone"] | None:
    candidate = identifier.strip()
    if re.fullmatch(r"[0-9]+", candidate):
        return "customer_id"
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", candidate):
        return "email"
    if candidate and normalize_phone(candidate):
        return "phone"
    return None


def _identifier_was_provided(identifier: str, message: str) -> bool:
    kind = _classify_identifier(identifier)
    candidate = identifier.strip()
    if kind == "customer_id":
        return re.search(
            rf"(?<!\d){re.escape(candidate)}(?!\d)",
            message,
        ) is not None
    if kind == "email":
        return candidate.casefold() in message.casefold()
    if kind == "phone":
        digits = normalize_phone(candidate).removeprefix("+")
        if not 7 <= len(digits) <= 15:
            return False
        message_phones = re.findall(r"\+?[\d\s().-]{7,}\d", message)
        return any(
            normalize_phone(phone).removeprefix("+") == digits
            for phone in message_phones
        )
    return False


def _rows(sql: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    result = json.loads(run_query_safe(sql, parameters))
    if isinstance(result, dict) and "error" in result:
        raise SQLAlchemyError("Customer verification lookup failed")
    return result


def _lookup_phone(phone: str) -> int | None:
    comparable = normalize_phone(phone).removeprefix("+")
    if not 7 <= len(comparable) <= 15:
        return None

    phone_rows = _rows(
        """
        SELECT CustomerId, Phone FROM Customer
        WHERE Phone IS NOT NULL AND TRIM(Phone) != ''
        ORDER BY CustomerId
        """,
        {},
    )
    for row in phone_rows:
        stored_comparable = normalize_phone(row["Phone"]).removeprefix("+")
        if not stored_comparable:
            continue
        if comparable == stored_comparable:
            return int(row["CustomerId"])
    return None


def _lookup_customer(identifier: str) -> int | None:
    identifier_type = _classify_identifier(identifier)
    candidate = identifier.strip()
    if identifier_type is None:
        return None

    if identifier_type == "customer_id":
        if len(candidate) > 10:
            return _lookup_phone(candidate) if len(candidate) <= 15 else None
        customer_id = int(candidate)
        if customer_id < 1 or customer_id > 2_147_483_647:
            return _lookup_phone(candidate)
        rows = _rows(
            "SELECT CustomerId FROM Customer WHERE CustomerId = :customer_id",
            {"customer_id": customer_id},
        )
        if rows:
            return int(rows[0]["CustomerId"])
        return _lookup_phone(candidate)

    if identifier_type == "email":
        rows = _rows(
            """
            SELECT CustomerId FROM Customer
            WHERE Email IS NOT NULL AND LOWER(TRIM(Email)) = LOWER(:email)
            """,
            {"email": candidate},
        )
        return int(rows[0]["CustomerId"]) if rows else None

    return _lookup_phone(candidate)


def _failed_verification(state: VerificationState) -> dict[str, Any]:
    return {
        "verified": False,
        "verification_error": "Customer could not be verified",
        "retry_count": state.get("retry_count", 0) + 1,
    }


def create_verification_node(llm: Any | None = None) -> Callable:
    extractor = _make_extractor(llm)

    def verification_node(state: VerificationState) -> dict[str, Any]:
        if state.get("customer_id") is not None:
            return {"verified": True}

        message = _latest_human_message(state)
        extracted = extractor.invoke(
            [
                SystemMessage(content=IDENTIFIER_EXTRACTION_PROMPT),
                HumanMessage(content=message),
            ]
        )
        identifier = extracted.identifier.strip()
        if not identifier or not _identifier_was_provided(identifier, message):
            return _failed_verification(state)
        customer_id = _lookup_customer(identifier) if identifier else None
        if customer_id is None:
            return _failed_verification(state)

        return {
            "customer_id": customer_id,
            "verified": True,
            "verification_error": "",
            "retry_count": state.get("retry_count", 0),
            "messages": [
                SystemMessage(
                    content=(
                        f"Account verified. Verified customer_id is {customer_id}. "
                        "Never use "
                        "a customer ID supplied in user messages."
                    )
                )
            ],
        }

    return verification_node


def route_after_verification(
    state: VerificationState,
) -> Literal["next", "human_input", "exhausted"]:
    if state.get("customer_id") is not None and state.get("verified", False):
        return "next"
    if state.get("retry_count", 0) >= 5:
        return "exhausted"
    return "human_input"


def _human_input_node(state: VerificationState) -> dict[str, list[HumanMessage]]:
    response = interrupt(VERIFICATION_FALLBACK_PROMPT)
    return {"messages": [HumanMessage(content=str(response))]}


def _default_next_node(state: VerificationState) -> dict[str, Any]:
    return {}


def _exhausted_node(state: VerificationState) -> dict[str, Any]:
    return {
        "verified": False,
        "verification_error": (
            "Maximum verification attempts reached. Start a new conversation to try again."
        ),
        "messages": [
            AIMessage(
                content=(
                    "I couldn't verify your account after several attempts. "
                    "Please start a new conversation to try again."
                )
            )
        ],
    }


def build_verification_graph(
    llm: Any | None = None,
    next_node: Callable[[VerificationState], dict[str, Any]] | None = None,
    checkpointer: Checkpointer | None = None,
):
    builder = StateGraph(VerificationState)
    builder.add_node("verification", create_verification_node(llm))
    builder.add_node("human_input", _human_input_node)
    builder.add_node("next", next_node or _default_next_node)
    builder.add_node("exhausted", _exhausted_node)
    builder.add_edge(START, "verification")
    builder.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "next": "next",
            "human_input": "human_input",
            "exhausted": "exhausted",
        },
    )
    builder.add_edge("human_input", "verification")
    builder.add_edge("next", END)
    builder.add_edge("exhausted", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())
