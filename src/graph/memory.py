import json
import logging
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from src.config import settings
from src.config.logging import log_event

MEMORY_NAMESPACE = "memory_profile"
MEMORY_KEY = "user_memory"
PREFERENCE_CONTEXT_MESSAGE_ID = "saved-music-preferences"
logger = logging.getLogger(__name__)

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
        if isinstance(content, str) and _is_explicit_positive_preference(content):
            lines.append(f"Customer: {content.strip()}")
    return "\n".join(lines)


def _is_explicit_positive_preference(message: str) -> bool:
    text = message.strip()
    if not text or "?" in text:
        return False
    if re.search(
        r"\b(?:not|never|no|don't|doesn't|didn't|dislike|hate|"
        r"if|would|could|might|suppose|hypothetical)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:friend|wife|husband|partner|mother|father|brother|sister|"
        r"son|daughter|colleague|coworker|someone else|other people)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.match(
            r"^(?:i\s+(?:(?:really|absolutely|particularly)\s+)?"
            r"(?:love|like|prefer|enjoy|adore|am\s+a\s+fan\s+of)\b|"
            r"my\s+(?:favorite|favourite)\b|"
            r"[\w'&./ -]+\s+is\s+my\s+(?:favorite|favourite)\b)",
            text,
            re.IGNORECASE,
        )
    )


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
        try:
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
        except Exception as error:  # noqa: BLE001 - extraction failures must not block replies
            log_event(
                logger,
                "memory.extraction_failed",
                level=logging.WARNING,
                error_type=type(error).__name__,
                outcome="skipped",
            )
            return {}
        if profile.customer_id != customer_id:
            log_event(
                logger,
                "memory.extraction_rejected",
                level=logging.WARNING,
                outcome="customer_id_mismatch",
            )
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
            log_event(
                logger,
                "memory.preferences_updated",
                preference_count=len(merged),
                outcome="success",
            )
        return {
            "music_preferences": merged,
            "music_preferences_context": _preferences_context(merged),
        }

    return create_memory


class PersistentPreferenceStore(InMemoryStore):
    """Keep LangGraph's existing store API while persisting user profiles in SQLite.

    The educational app uses namespace/key put/get for customer preferences.
    This implementation persists those calls without adding another package.
    A single container/process owns the store; use an external managed database
    instead when deploying multiple replicas.
    """

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.RLock()
        self._db = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            timeout=30,
        )
        self._db.execute("PRAGMA busy_timeout = 30000")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS preference_profiles (
            customer_id TEXT NOT NULL PRIMARY KEY,
            document TEXT NOT NULL
        )""")
        self._db.commit()
        for customer_id, raw_value in self._db.execute(
            "SELECT customer_id, document FROM preference_profiles"
        ).fetchall():
            try:
                value = json.loads(raw_value)
                if isinstance(value, dict):
                    super().put((MEMORY_NAMESPACE, customer_id), MEMORY_KEY, value)
            except (ValueError, TypeError):
                continue

    def put(self, namespace, key, value, index=None):
        # Save first, then update the in-memory index, so a failed write is not
        # falsely presented as a persisted preference.
        if (len(namespace) == 2 and namespace[0] == MEMORY_NAMESPACE
                and key == MEMORY_KEY):
            with self._db_lock:
                self._db.execute(
                    """INSERT INTO preference_profiles (customer_id, document)
                       VALUES (?, ?)
                       ON CONFLICT(customer_id) DO UPDATE SET document=excluded.document""",
                    (str(namespace[1]), json.dumps(value)),
                )
                self._db.commit()
                return super().put(namespace, key, value, index=index)
        return super().put(namespace, key, value, index=index)

    def close(self) -> None:
        with self._db_lock:
            self._db.close()


def create_memory_store() -> InMemoryStore:
    # Persist customer profiles in the app data directory or configured volume.
    path = os.getenv("MEMORY_DB_PATH") or str(
        Path(__file__).resolve().parents[2] / "app" / "data" / "preferences.sqlite3"
    )
    return PersistentPreferenceStore(path)
