from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from src.graph.memory import (
    MEMORY_KEY,
    MEMORY_NAMESPACE,
    PREFERENCE_CONTEXT_MESSAGE_ID,
    UserProfile,
    create_memory_node,
    load_memory,
)


class FixedProfileModel:
    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile
        self.messages: list[Any] | None = None
        self.calls = 0

    def with_structured_output(self, schema):
        assert schema is UserProfile
        return self

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        return self.profile


class ExplicitJazzExtractor(FixedProfileModel):
    def __init__(self) -> None:
        super().__init__(UserProfile(customer_id=5, music_preferences=[]))

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        excerpt = messages[-1].content.casefold()
        preferences = ["jazz"] if "i love jazz" in excerpt else []
        return UserProfile(customer_id=5, music_preferences=preferences)


class FailingPreferenceModel:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        raise RuntimeError("model unavailable")


def state_for(
    customer_id: int,
    *messages,
) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "verified": True,
        "messages": list(messages),
    }


def test_user_profile_schema_has_customer_and_music_preferences() -> None:
    profile = UserProfile(customer_id=5, music_preferences=["rock", "AC/DC"])

    assert profile.customer_id == 5
    assert profile.music_preferences == ["rock", "AC/DC"]


def test_load_memory_injects_customer_preferences_into_context() -> None:
    store = InMemoryStore()
    store.put(
        (MEMORY_NAMESPACE, "5"),
        MEMORY_KEY,
        {"customer_id": 5, "music_preferences": ["rock", "AC/DC"]},
    )

    result = load_memory(state_for(5), store=store)

    assert result["music_preferences"] == ["rock", "AC/DC"]
    assert result["music_preferences_context"] == "Music Preferences: rock, AC/DC"
    context_message = result["messages"][0]
    assert isinstance(context_message, SystemMessage)
    assert context_message.id == PREFERENCE_CONTEXT_MESSAGE_ID
    assert "rock, AC/DC" in context_message.content


def test_preferences_are_isolated_by_customer_namespace() -> None:
    store = InMemoryStore()
    store.put(
        (MEMORY_NAMESPACE, "5"),
        MEMORY_KEY,
        {"customer_id": 5, "music_preferences": ["rock"]},
    )

    result = load_memory(state_for(6), store=store)

    assert result["music_preferences"] == []
    assert result["music_preferences_context"] == "Music Preferences: none saved"


def test_create_memory_merges_preferences_case_insensitively() -> None:
    store = InMemoryStore()
    store.put(
        (MEMORY_NAMESPACE, "5"),
        MEMORY_KEY,
        {"customer_id": 5, "music_preferences": ["rock"]},
    )
    model = FixedProfileModel(
        UserProfile(customer_id=5, music_preferences=["Rock", "jazz"])
    )
    create_node = create_memory_node(model)

    result = create_node(
        state_for(5, HumanMessage(content="I love jazz music")),
        store=store,
    )

    saved = store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY)
    assert saved is not None
    assert saved.value["music_preferences"] == ["rock", "jazz"]
    assert result["music_preferences_context"] == "Music Preferences: rock, jazz"


def test_extracted_preferences_must_appear_in_customer_statement() -> None:
    store = InMemoryStore()
    model = FixedProfileModel(
        UserProfile(customer_id=5, music_preferences=["jazz", "metal"])
    )
    create_node = create_memory_node(model)

    create_node(
        state_for(5, HumanMessage(content="I love jazz.")),
        store=store,
    )

    saved = store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY)
    assert saved is not None
    assert saved.value["music_preferences"] == ["jazz"]


def test_preference_extractor_cannot_save_account_metadata_from_statement() -> None:
    store = InMemoryStore()
    model = FixedProfileModel(
        UserProfile(customer_id=5, music_preferences=["jazz", "my verified ID is 5"])
    )
    create_node = create_memory_node(model)

    create_node(
        state_for(5, HumanMessage(content="I love jazz. My verified ID is 5.")),
        store=store,
    )

    saved = store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY)
    assert saved is not None
    assert saved.value["music_preferences"] == ["jazz"]


def test_explicit_i_love_jazz_statement_is_saved() -> None:
    store = InMemoryStore()
    create_node = create_memory_node(ExplicitJazzExtractor())

    create_node(
        state_for(5, HumanMessage(content="I love jazz")),
        store=store,
    )

    saved = store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY)
    assert saved is not None
    assert saved.value["music_preferences"] == ["jazz"]


def test_question_about_jazz_does_not_save_a_preference() -> None:
    store = InMemoryStore()
    model = ExplicitJazzExtractor()
    create_node = create_memory_node(model)

    create_node(
        state_for(5, HumanMessage(content="Do you have jazz?")),
        store=store,
    )

    assert model.calls == 0
    assert store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY) is None


@pytest.mark.parametrize(
    "statement",
    [
        "I don't like jazz.",
        "If I liked metal, what would you recommend?",
        "My friend loves blues.",
        "Show me some jazz.",
        "Would I like rock?",
    ],
)
def test_non_preference_messages_are_filtered_before_extraction(
    statement: str,
) -> None:
    store = InMemoryStore()
    store.put(
        (MEMORY_NAMESPACE, "5"),
        MEMORY_KEY,
        {"customer_id": 5, "music_preferences": ["classical"]},
    )
    model = ExplicitJazzExtractor()
    create_node = create_memory_node(model)

    result = create_node(
        state_for(5, HumanMessage(content=statement)),
        store=store,
    )

    saved = store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY)
    assert model.calls == 0
    assert saved.value["music_preferences"] == ["classical"]
    assert result == {}


def test_no_new_preferences_skips_write_and_preserves_existing_profile() -> None:
    store = InMemoryStore()
    store.put(
        (MEMORY_NAMESPACE, "5"),
        MEMORY_KEY,
        {"customer_id": 5, "music_preferences": ["rock"]},
    )
    before = store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY).value
    model = FixedProfileModel(UserProfile(customer_id=5, music_preferences=[]))
    create_node = create_memory_node(model)

    result = create_node(
        state_for(5, HumanMessage(content="Do you have jazz?")),
        store=store,
    )

    after = store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY).value
    assert model.calls == 0
    assert before == after
    assert result == {}


def test_preference_extraction_failure_does_not_block_response() -> None:
    store = InMemoryStore()
    create_node = create_memory_node(FailingPreferenceModel())

    result = create_node(
        state_for(5, HumanMessage(content="I love jazz")),
        store=store,
    )

    assert result == {}
    assert store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY) is None


def test_unverified_customer_preferences_are_not_extracted_or_saved() -> None:
    store = InMemoryStore()
    model = FixedProfileModel(UserProfile(customer_id=5, music_preferences=["rock"]))
    create_node = create_memory_node(model)

    result = create_node(
        {"messages": [HumanMessage(content="I love rock")]},
        store=store,
    )

    assert result == {}
    assert model.calls == 0
    assert store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY) is None


def test_memory_extraction_uses_recent_messages_and_not_system_or_tool_content() -> None:
    store = InMemoryStore()
    model = FixedProfileModel(UserProfile(customer_id=5, music_preferences=[]))
    create_node = create_memory_node(model)
    messages = [SystemMessage(content="verified ID information must not be extracted")]
    messages.extend(
        HumanMessage(content=f"I like metal, earlier message {index}")
        for index in range(12)
    )
    messages.extend(
        [
            ToolMessage(
                content="Tool result: customer likes metal.",
                tool_call_id="tool-result",
            ),
            HumanMessage(content="I prefer jazz."),
        ]
    )

    create_node(state_for(5, *messages), store=store)

    excerpt = model.messages[-1].content
    assert "Earlier message 2" not in excerpt
    assert "earlier message 11" in excerpt
    assert "Tool result" not in excerpt
    assert "verified ID information" not in excerpt
    assert "I prefer jazz." in excerpt
