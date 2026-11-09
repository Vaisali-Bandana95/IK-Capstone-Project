"""Acceptance regressions for the required capstone conversation paths.

No live LLM credentials are required. The real SQL fixture comes from conftest.
"""
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from src.agents.supervisor import build_support_graph
from src.graph.memory import (
    MEMORY_KEY,
    MEMORY_NAMESPACE,
    PersistentPreferenceStore,
    UserProfile,
)
from src.tools.invoice import get_invoices_by_customer_sorted_by_date
from src.tools.music import get_albums_by_artist, get_songs_by_genre


class ExtractedID:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        text = str(messages[-1].content)
        # Contract tests are for the graph, not the LLM's ID extraction accuracy.
        import re
        match = re.search(r"\b\d+\b", text)
        return type("Output", (), {"identifier": match.group() if match else ""})()


class EmptyPreferenceModel:
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return UserProfile(customer_id=5, music_preferences=[])


class RecordingAgent:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def invoke(self, input, config=None):
        query = input["messages"][-1].content
        self.calls.append((self.name, query, config))
        return {"messages": [AIMessage(content=f"{self.name} grounded result")]}


def build_offline_graph(calls, store=None):
    return build_support_graph(
        model=object(),
        verification_llm=ExtractedID(),
        checkpointer=MemorySaver(),
        music_agent=RecordingAgent("music", calls),
        invoice_agent=RecordingAgent("invoice", calls),
        memory_model=EmptyPreferenceModel(),
        store=store or InMemoryStore(),
    )


def config(suffix):
    return {"configurable": {"thread_id": f"acceptance-{suffix}"}}


def test_off_topic_skips_identity_and_both_specialists():
    calls = []
    graph = build_offline_graph(calls)
    result = graph.invoke({"messages": [HumanMessage(content="What's the weather today?")]}, config=config("weather"))
    assert "unrelated topics" in result["messages"][-1].content
    assert calls == []
    assert not result.get("verified")


def test_catalog_is_accessible_without_invoice_identity():
    calls = []
    graph = build_offline_graph(calls)
    result = graph.invoke({"messages": [HumanMessage(content="What rock albums do you have?")]}, config=config("catalog"))
    assert calls[0][0] == "music"
    assert not result.get("verified")
    assert "music grounded result" in result["messages"][-1].content


def test_invoice_requires_verification_and_does_not_lose_original_question():
    calls = []
    graph = build_offline_graph(calls)
    state = graph.invoke({"messages": [HumanMessage(content="Show me my recent purchases")]}, config=config("invoice"))
    assert not calls
    assert not state.get("verified")
    state = graph.invoke(Command(resume="5"), config=config("invoice"))
    assert state["verified"] is True
    assert calls[0][0] == "invoice"
    assert calls[0][1] == "Show me my recent purchases"
    assert calls[0][2]["configurable"]["verified_customer_id"] == 5


def test_purchased_tracks_question_routes_to_invoice_agent():
    calls = []
    graph = build_offline_graph(calls)
    thread_config = config("purchased-tracks")

    state = graph.invoke(
        {"messages": [HumanMessage(content="what tracks have I purchased?")]},
        config=thread_config,
    )

    assert not state.get("verified")
    assert calls == []

    state = graph.invoke(Command(resume="5"), config=thread_config)

    assert state["verified"] is True
    assert len(calls) == 1
    assert calls[0][0] == "invoice"
    assert calls[0][1] == "what tracks have I purchased?"
    assert calls[0][2]["configurable"]["verified_customer_id"] == 5


def test_genre_question_about_purchased_track_routes_only_to_catalog():
    calls = []
    graph = build_offline_graph(calls)
    request = 'What is the genre of the song "Stir It Up (Live)" I purchased?'
    state = graph.invoke(
        {"messages": [HumanMessage(content=request)]},
        config=config("purchased-track-genre"),
    )

    assert len(calls) == 1
    assert calls[0][0] == "music"
    assert calls[0][1] == request
    assert state["route"] == "music"
    assert not state.get("verified")
    assert "Music catalog:" not in state["messages"][-1].content
    assert "Your purchases:" not in state["messages"][-1].content


def test_purchase_list_and_genre_question_routes_to_both_specialists():
    calls = []
    graph = build_offline_graph(calls)
    thread_config = config("purchased-tracks-and-genre")
    request = 'What songs have I purchased, and what genre is "Stir It Up (Live)"?'

    graph.invoke(
        {"messages": [HumanMessage(content=request)]},
        config=thread_config,
    )
    state = graph.invoke(Command(resume="5"), config=thread_config)

    assert [call[0] for call in calls] == ["invoice", "music"]
    assert "Your purchases:" in state["messages"][-1].content
    assert "Music catalog:" in state["messages"][-1].content


def test_purchased_tracks_with_catalog_recommendation_routes_to_both_agents():
    calls = []
    graph = build_offline_graph(calls)
    thread_config = config("purchased-tracks-and-recommendation")
    request = "Which songs did I buy? Also recommend jazz albums."

    graph.invoke(
        {"messages": [HumanMessage(content=request)]},
        config=thread_config,
    )
    state = graph.invoke(Command(resume="5"), config=thread_config)

    assert [call[0] for call in calls] == ["invoice", "music"]
    assert all(call[1] == request for call in calls)
    assert "Your purchases:" in state["messages"][-1].content
    assert "Music catalog:" in state["messages"][-1].content


def test_mixed_request_calls_invoice_first_and_music_second():
    calls = []
    graph = build_offline_graph(calls)
    graph.invoke({"messages": [HumanMessage(content="Show my invoices and what jazz albums you have")]}, config=config("mixed"))
    state = graph.invoke(Command(resume="5"), config=config("mixed"))
    assert [item[0] for item in calls] == ["invoice", "music"]
    assert all("jazz albums" in item[1] for item in calls)
    assert "invoice grounded result" in state["messages"][-1].content
    assert "music grounded result" in state["messages"][-1].content


def test_invalid_id_cannot_trigger_invoice_agent():
    calls = []
    graph = build_offline_graph(calls)
    state = graph.invoke({"messages": [HumanMessage(content="My customer ID is 99999")]}, config=config("invalid"))
    assert not state.get("verified")
    assert not calls


def test_formatted_phone_can_start_verification_and_resume():
    calls = []
    graph = build_offline_graph(calls)
    thread_config = config("formatted-phone")

    state = graph.invoke(
        {"messages": [HumanMessage(content="Show my invoices")]},
        config=thread_config,
    )

    assert not state.get("verified")
    snapshot = graph.get_state(thread_config)
    assert snapshot.next == ("human_input",)
    assert "Customer ID, email address, or phone number" in (
        snapshot.tasks[0].interrupts[0].value
    )

    state = graph.invoke(
        Command(resume="+55 (12) 3923-5555"),
        config=thread_config,
    )

    assert state["verified"] is True
    assert state["customer_id"] == 1
    assert calls[0][0] == "invoice"
    assert calls[0][2]["configurable"]["verified_customer_id"] == 1


def test_unknown_id_in_existing_checkpoint_does_not_reuse_previous_customer():
    calls = []
    graph = build_offline_graph(calls)
    thread_config = config("stale-verified-customer")
    graph.update_state(
        thread_config,
        {
            "customer_id": 12,
            "verified": True,
            "messages": [
                AIMessage(
                    content="Account lookup successful for customer ID 12."
                )
            ],
        },
    )

    state = graph.invoke(
        {"messages": [HumanMessage(content="My ID is 99999")]},
        config=thread_config,
    )

    assert state["verified"] is False
    assert state["customer_id"] == 12
    assert calls == []
    snapshot = graph.get_state(thread_config)
    assert snapshot.next == ("human_input",)
    assert "couldn't find an account" in snapshot.tasks[0].interrupts[0].value


def test_genre_count_is_tracks_not_distinct_artists():
    from src.db import execute_query
    answer = json.loads(get_songs_by_genre.invoke({"genre": "Rock"}))
    rows = json.loads(execute_query('''
        SELECT COUNT(*) AS tracks, COUNT(DISTINCT Album.AlbumId) AS albums
        FROM Track JOIN Album ON Album.AlbumId = Track.AlbumId
        JOIN Genre ON Genre.GenreId = Track.GenreId
        WHERE LOWER(Genre.Name) LIKE :genre
    ''', {"genre": "%rock%"}))
    assert answer["total_count"] == rows[0]["tracks"]
    assert answer["album_count"] == rows[0]["albums"]
    assert len({row["artist"] for row in answer["tracks"]}) == len(answer["tracks"])


def test_invoice_total_is_database_aggregated():
    from src.db import execute_query
    answers = json.loads(get_invoices_by_customer_sorted_by_date.invoke({"customer_id": "5"}))
    expected = json.loads(execute_query(
        "SELECT printf('%.2f', SUM(Total)) AS value FROM Invoice WHERE CustomerId = :customer_id",
        {"customer_id": 5},
    ))[0]["value"]
    assert answers and all(row["total_spent"] == expected for row in answers)


def test_preferences_survive_new_store_instance(tmp_path: Path):
    path = tmp_path / "prefs.sqlite3"
    first = PersistentPreferenceStore(path)
    first.put((MEMORY_NAMESPACE, "5"), MEMORY_KEY,
              {"customer_id": 5, "music_preferences": ["jazz", "AC/DC"]})
    second = PersistentPreferenceStore(path)
    item = second.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY)
    assert item.value["music_preferences"] == ["jazz", "AC/DC"]


def test_artist_typo_has_fuzzy_match():
    albums = json.loads(get_albums_by_artist.invoke({"artist_name": "AC DC"}))
    assert isinstance(albums, list) and albums
    assert any(row["artist"] == "AC/DC" for row in albums)
