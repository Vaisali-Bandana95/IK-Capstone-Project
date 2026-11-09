import json
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

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
ACCOUNT_NOT_FOUND_PROMPT = (
    "I couldn't find an account matching that identifier. Please try a different "
    "Customer ID, email address, or phone number."
)
ACCOUNT_SWITCH_PROMPT = (
    "This conversation is already associated with a different account. Start a "
    "new conversation to verify another account."
)
AMBIGUOUS_IDENTIFIER_PROMPT = (
    "I found multiple identifiers in that message. Please send one Customer ID, "
    "email address, or phone number at a time."
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


def extract_phone_identifier(message: str) -> str | None:
    phone = re.search(
        r"\b(?:phone|mobile|telephone|cell)(?:\s+number)?\s*"
        r"(?:is|[:=])?\s*(\+?\d[\d\s().-]*\d)",
        message,
        re.IGNORECASE,
    )
    if phone:
        candidate = phone.group(1).strip()
        if 7 <= len(normalize_phone(candidate).removeprefix("+")) <= 15:
            return candidate

    if re.fullmatch(r"\s*\+?\d[\d\s().-]*\d\s*", message):
        candidate = message.strip()
        if 7 <= len(normalize_phone(candidate).removeprefix("+")) <= 15:
            return candidate
    return None


def _extract_identifier(message: str, extractor: Runnable) -> str:
    explicit_customer_id = re.search(
        r"\b(?:my\s+)?(?:customer\s+)?id\s*(?:is|[:=])?\s*([0-9]+)\b",
        message,
        re.IGNORECASE,
    )
    if explicit_customer_id:
        return explicit_customer_id.group(1)

    if re.fullmatch(r"\s*[0-9]+\s*", message):
        return message.strip()

    email = re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+", message)
    if email:
        return email.group()

    phone = extract_phone_identifier(message)
    if phone:
        return phone

    extracted = extractor.invoke(
        [
            SystemMessage(content=IDENTIFIER_EXTRACTION_PROMPT),
            HumanMessage(content=message),
        ]
    )
    return extracted.identifier.strip()


def _contains_multiple_identifiers(message: str) -> bool:
    candidates: set[tuple[str, str]] = set()
    customer_ids = re.findall(
        r"\b(?:my\s+)?(?:customer\s+)?id\s*(?:is|[:=])?\s*([0-9]+)\b",
        message,
        re.IGNORECASE,
    )
    candidates.update(("customer_id", value) for value in customer_ids)
    if (
        not customer_ids
        and re.fullmatch(r"\s*[0-9]+\s*", message)
        and len(message.strip()) <= 10
    ):
        candidates.add(("customer_id", message.strip()))
    candidates.update(
        ("email", email.casefold())
        for email in re.findall(r"[^@\s]+@[^@\s]+\.[^@\s]+", message)
    )
    phone = extract_phone_identifier(message)
    if phone:
        candidates.add(("phone", normalize_phone(phone).removeprefix("+")))
    return len(candidates) > 1


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
    normalized_rows = [
        (
            int(row["CustomerId"]),
            normalize_phone(row["Phone"]).removeprefix("+"),
        )
        for row in phone_rows
    ]
    exact_matches = [
        customer_id
        for customer_id, stored_phone in normalized_rows
        if stored_phone and comparable == stored_phone
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if exact_matches:
        return None

    if len(comparable) >= 10:
        last_ten_matches = [
            customer_id
            for customer_id, stored_phone in normalized_rows
            if len(stored_phone) >= 10
            and comparable[-10:] == stored_phone[-10:]
        ]
        if len(last_ten_matches) == 1:
            return last_ten_matches[0]
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


def _failed_verification(
    state: VerificationState,
    message: str = VERIFICATION_FALLBACK_PROMPT,
) -> dict[str, Any]:
    return {
        "verified": False,
        "verification_error": message,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def create_verification_node(llm: Any | None = None) -> Callable:
    extractor = _make_extractor(llm)

    def verification_node(state: VerificationState) -> dict[str, Any]:
        message = _latest_human_message(state)
        if _contains_multiple_identifiers(message):
            return _failed_verification(state, AMBIGUOUS_IDENTIFIER_PROMPT)
        identifier = _extract_identifier(message, extractor)
        if not identifier or not _identifier_was_provided(identifier, message):
            return _failed_verification(state)
        customer_id = _lookup_customer(identifier) if identifier else None
        if customer_id is None:
            return _failed_verification(state, ACCOUNT_NOT_FOUND_PROMPT)
        existing_customer_id = state.get("customer_id")
        if (
            existing_customer_id is not None
            and customer_id != existing_customer_id
        ):
            return _failed_verification(state, ACCOUNT_SWITCH_PROMPT)

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
    response = interrupt(
        state.get("verification_error") or VERIFICATION_FALLBACK_PROMPT
    )
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
