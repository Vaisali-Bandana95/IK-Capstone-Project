from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from src.config import settings

MEMORY_NAMESPACE = "memory_profile"
MEMORY_KEY = "user_memory"
PREFERENCE_CONTEXT_MESSAGE_ID = "saved-music-preferences"

MEMORY_EXTRACTION_PROMPT = """Extract only explicit music preferences stated by the customer in their own messages.
Save direct, positive preference statements such as “I love rock”, “AC/DC is my
favorite band”, or “I prefer jazz”. Do not save questions (“Do you have rock?”),
requests, hypotheticals, negatives or dislikes, uncertain statements, third-party
preferences, assistant statements, or facts from tool results. The supplied
excerpt contains only the customer's most recent messages.
Return the verified customer_id supplied in the instructions and a list containing
only newly stated preference phrases. If there are no new explicit preferences,
return an empty list. Never infer a preference from a question or a search request."""


class UserProfile(BaseModel):
    customer_id: int = Field(gt=0)
    music_preferences: list[str] = Field(default_factory=list)


class MemoryState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: int
    verified: bool
    music_preferences: list[str]
    music_preferences_context: str


def _namespace(customer_id: int) -> tuple[str, str]:
    return (MEMORY_NAMESPACE, str(customer_id))


def _stored_preferences(store: BaseStore, customer_id: int) -> list[str]:
    item = store.get(_namespace(customer_id), MEMORY_KEY)
    if item is None:
        return []

    raw_preferences = item.value.get("music_preferences", [])
    if not isinstance(raw_preferences, list):
        return []
    return [
        preference.strip()
        for preference in raw_preferences
        if isinstance(preference, str) and preference.strip()
    ]


def _preferences_context(preferences: list[str]) -> str:
    if not preferences:
        return "Music Preferences: none saved"
    return f"Music Preferences: {', '.join(preferences)}"


def load_memory(state: MemoryState, *, store: BaseStore) -> dict[str, Any]:
    customer_id = state.get("customer_id")
    if not state.get("verified") or customer_id is None:
        return {
            "music_preferences": [],
            "music_preferences_context": "Music Preferences: unavailable until verified",
        }

    preferences = _stored_preferences(store, customer_id)
    context = _preferences_context(preferences)
    return {
        "music_preferences": preferences,
        "music_preferences_context": context,
        "messages": [
            SystemMessage(
                id=PREFERENCE_CONTEXT_MESSAGE_ID,
                content=(
                    f"{context}. Use these preferences only to personalize "
                    "catalog suggestions; factual catalog claims must come from "
                    "music tools."
                ),
            )
        ],
    }


def _conversation_excerpt(messages: list[BaseMessage]) -> str:
    customer_messages = [
        message for message in messages if isinstance(message, HumanMessage)
    ][-10:]
    lines = []
    for message in customer_messages:
        content = message.content
        if isinstance(content, str) and content.strip():
            lines.append(f"Customer: {content.strip()}")
    return "\n".join(lines)


def _preference_extractor(model=None):
    if model is None:
        if not settings.api_key:
            raise ValueError("OPENAI_API_KEY must be configured for memory extraction")
        if not settings.model_name:
            raise ValueError("OPENAI_MODEL must be configured for memory extraction")

        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            api_key=settings.api_key,
            model=settings.model_name,
            temperature=0,
            base_url=settings.api_base_url,
        )
    return model.with_structured_output(UserProfile)


def create_memory_node(model=None):
    extractor = None
    if model is not None:
        extractor = _preference_extractor(model)

    def create_memory(state: MemoryState, *, store: BaseStore) -> dict[str, Any]:
        customer_id = state.get("customer_id")
        if not state.get("verified") or customer_id is None:
            return {}

        existing = _stored_preferences(store, customer_id)
        excerpt = _conversation_excerpt(state.get("messages", []))
        if not excerpt:
            return {}

        active_extractor = extractor or _preference_extractor()
        profile = active_extractor.invoke(
            [
                SystemMessage(
                    content=(
                        f"{MEMORY_EXTRACTION_PROMPT}\n"
                        f"The verified customer_id is {customer_id}."
                    )
                ),
                HumanMessage(content=excerpt),
            ]
        )
        if profile.customer_id != customer_id:
            return {}

        new_preferences = [
            preference.strip()
            for preference in profile.music_preferences
            if preference.strip()
        ]
        if not new_preferences:
            return {}

        merged = list(existing)
        known = {preference.casefold() for preference in existing}
        for preference in new_preferences:
            folded = preference.casefold()
            if folded not in known:
                merged.append(preference)
                known.add(folded)

        if merged != existing:
            store.put(
                _namespace(customer_id),
                MEMORY_KEY,
                {
                    "customer_id": customer_id,
                    "music_preferences": merged,
                },
            )
        return {
            "music_preferences": merged,
            "music_preferences_context": _preferences_context(merged),
        }

    return create_memory


def create_memory_store() -> InMemoryStore:
    return InMemoryStore()
