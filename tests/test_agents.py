import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import PrivateAttr

from src.agents.invoice_agent import build_invoice_agent, build_invoice_tools
from src.agents.music_agent import MUSIC_TOOLS, build_music_agent
from src.agents.prompts import (
    INVOICE_AGENT_PROMPT,
    MUSIC_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
)
from src.agents.supervisor import (
    _specialist_result,
    build_supervisor,
    build_support_graph,
)
from src.graph import UserInput
from src.graph.memory import MEMORY_KEY, MEMORY_NAMESPACE, UserProfile


class EchoIdentifierLLM:
    def with_structured_output(self, schema: type[UserInput]) -> "EchoIdentifierLLM":
        return self

    def invoke(self, messages: list[Any]) -> UserInput:
        return UserInput(identifier=str(messages[-1].content))


class EmptyPreferenceLLM:
    def with_structured_output(self, schema):
        assert schema is UserProfile
        return self

    def invoke(self, messages):
        customer_id = int(messages[0].content.rsplit(" ", 1)[-1].rstrip("."))
        return UserProfile(customer_id=customer_id, music_preferences=[])


class JazzPreferenceLLM:
    def with_structured_output(self, schema):
        assert schema is UserProfile
        return self

    def invoke(self, messages):
        customer_id = int(messages[0].content.rsplit(" ", 1)[-1].rstrip("."))
        return UserProfile(customer_id=customer_id, music_preferences=["jazz"])


class RecordingSupervisor:
    def __init__(self) -> None:
        self.config = None
        self.inputs = []

    def invoke(self, input, config):
        self.config = config
        self.inputs.append(input)
        return {"messages": input["messages"]}


class ToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.received_messages = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        self.received_messages.append(messages)
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_albums_by_artist",
                        "args": {"artist_name": "AC/DC"},
                        "id": "album-search",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="AC/DC has albums in the catalog.")


class AlbumTrackListModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_tracks_by_album",
                        "args": {"album_title": "Into the Light"},
                        "id": "album-tracks",
                        "type": "tool_call",
                    }
                ],
            )

        tool_result = next(
            message
            for message in reversed(messages)
            if isinstance(message, ToolMessage)
            and message.name == "get_tracks_by_album"
        )
        tracks = json.loads(tool_result.content)["tracks"]
        return AIMessage(
            content="\n".join(
                f"- {track['title']} — {track['artist']}" for track in tracks
            )
        )


class DeterministicSupervisorModel(BaseChatModel):
    _bound_tools: list[Any] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "deterministic-supervisor-test"

    def bind_tools(self, tools, **kwargs):
        self._bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        request = next(
            message.content.lower()
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        )
        completed_handoffs = {
            message.name
            for message in messages
            if isinstance(message, ToolMessage)
            and message.content.startswith("Successfully transferred to ")
        }

        requested_agents = []
        if any(word in request for word in ("invoice", "billing", "purchase")):
            requested_agents.append("invoice_agent")
        if any(word in request for word in ("album", "artist", "song", "genre", "track")):
            requested_agents.append("music_agent")

        next_agent = next(
            (
                agent
                for agent in requested_agents
                if f"transfer_to_{agent}" not in completed_handoffs
            ),
            None,
        )
        if next_agent is None:
            content = (
                "I can help with music catalog and invoice questions."
                if not requested_agents
                else "Here is the grounded specialist result."
            )
            message = AIMessage(content=content)
        else:
            handoff_name = f"transfer_to_{next_agent}"
            assert handoff_name in {tool.name for tool in self._bound_tools}
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": handoff_name,
                        "args": {},
                        "id": f"call-{next_agent}",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def recording_agent(name: str, calls: list[str]):
    builder = StateGraph(MessagesState)

    def record(state):
        calls.append(name)
        return {"messages": [AIMessage(content=f"{name} used its tools")]}

    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    return builder.compile(name=name)


def test_all_three_prompts_include_six_grounding_rules() -> None:
    expected_rules = (
        "Tool-only responses",
        "Quote numbers",
        "Be honest",
        "Stay within",
        "Never answer from memory",
        "Disclose sampled results",
    )

    for prompt in (MUSIC_AGENT_PROMPT, INVOICE_AGENT_PROMPT, SUPERVISOR_PROMPT):
        assert all(rule in prompt for rule in expected_rules)
    assert "invoice agent first, then the music agent" in SUPERVISOR_PROMPT
    assert "without calling either agent" in SUPERVISOR_PROMPT


def test_music_agent_runs_assistant_tool_assistant_loop() -> None:
    model = ToolCallingModel()
    agent = build_music_agent(model)

    result = agent.invoke({"messages": [HumanMessage(content="Albums by AC/DC?")]})

    assert len(MUSIC_TOOLS) == 6
    assert model.calls == 2
    assert "AC/DC has albums" in result["messages"][-1].content
    assert any(message.type == "tool" for message in result["messages"])


def test_music_agent_returns_all_tracks_for_an_album_request() -> None:
    model = AlbumTrackListModel()
    agent = build_music_agent(model)

    result = agent.invoke(
        {"messages": [HumanMessage(content="Share all songs from Into the Light")]}
    )

    answer = result["messages"][-1].content
    tool_result = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.name == "get_tracks_by_album"
    )
    tracks = json.loads(tool_result.content)["tracks"]
    assert model.calls == 2
    assert len(tracks) == 12
    assert all(track["title"] in answer for track in tracks)
    assert all(track["artist"] in answer for track in tracks)


def test_music_agent_prompt_includes_loaded_preference_context() -> None:
    model = ToolCallingModel()
    agent = build_music_agent(model)
    agent.invoke(
        {
            "messages": [
                SystemMessage(
                    id="saved-music-preferences",
                    content="Music Preferences: jazz",
                ),
                HumanMessage(content="Recommend something"),
            ]
        }
    )

    prompt = model.received_messages[0][0].content
    assert "Music Preferences: jazz" in prompt


def test_music_prompt_requires_full_album_track_lists() -> None:
    model = ToolCallingModel()
    agent = build_music_agent(model)
    agent.invoke(
        {"messages": [HumanMessage(content="List every track on Into the Light")]}
    )

    prompt = model.received_messages[0][0].content
    assert "get_tracks_by_album" in prompt
    assert "list every returned track title with its artist" in prompt
    assert "do not omit entries" in prompt


def test_music_agent_genre_tool_returns_real_album_data() -> None:
    result = json.loads(
        next(
            tool
            for tool in MUSIC_TOOLS
            if tool.name == "get_songs_by_genre"
        ).invoke({"genre": "Rock"})
    )

    assert result["tracks"]
    assert all(track["album"] for track in result["tracks"])
    assert any(track["genre"] == "Rock" for track in result["tracks"])


def test_supervisor_routes_music_queries_to_music_agent() -> None:
    calls: list[str] = []
    supervisor = build_supervisor(
        model=DeterministicSupervisorModel(),
        music_agent=recording_agent("music_agent", calls),
        invoice_agent=recording_agent("invoice_agent", calls),
    )

    result = supervisor.invoke(
        {"messages": [HumanMessage(content="What rock albums do you have?")]}
    )

    assert calls == ["music_agent"]
    assert "grounded specialist result" in result["messages"][-1].content


def test_supervisor_routes_invoice_queries_to_invoice_agent() -> None:
    calls: list[str] = []
    supervisor = build_supervisor(
        model=DeterministicSupervisorModel(),
        music_agent=recording_agent("music_agent", calls),
        invoice_agent=recording_agent("invoice_agent", calls),
    )

    supervisor.invoke({"messages": [HumanMessage(content="Show me my invoices")]})

    assert calls == ["invoice_agent"]


def test_supervisor_rejects_off_topic_without_subagent_calls() -> None:
    calls: list[str] = []
    supervisor = build_supervisor(
        model=DeterministicSupervisorModel(),
        music_agent=recording_agent("music_agent", calls),
        invoice_agent=recording_agent("invoice_agent", calls),
    )

    result = supervisor.invoke({"messages": [HumanMessage(content="What's the weather?")]})

    assert calls == []
    assert "I can help with music catalog and invoice questions." in (
        result["messages"][-1].content
    )


def test_supervisor_runs_mixed_request_invoice_then_music() -> None:
    calls: list[str] = []
    supervisor = build_supervisor(
        model=DeterministicSupervisorModel(),
        music_agent=recording_agent("music_agent", calls),
        invoice_agent=recording_agent("invoice_agent", calls),
    )

    supervisor.invoke(
        {
            "messages": [
                HumanMessage(content="Show my invoices and what jazz albums you have")
            ]
        }
    )

    assert calls == ["invoice_agent", "music_agent"]


def test_specialist_answer_without_tool_result_is_rejected() -> None:
    class UnsupportedAnswerAgent:
        def invoke(self, input, config=None):
            return {
                "messages": [
                    AIMessage(content="The album was released in 1980.")
                ]
            }

    response = _specialist_result(
        UnsupportedAnswerAgent(),
        {
            "messages": [HumanMessage(content="When was this album released?")],
            "current_request": "When was this album released?",
        },
        {},
        invoice=False,
    )

    assert "I won't guess" in response
    assert "catalog or account data" in response


def test_specialist_answer_after_tool_error_is_rejected() -> None:
    class ToolErrorAgent:
        def invoke(self, input, config=None):
            return {
                "messages": [
                    ToolMessage(
                        content='{"error":"query failed","code":"QUERY_ERROR"}',
                        tool_call_id="failed-tool",
                    ),
                    AIMessage(content="There are 10,000 albums."),
                ]
            }

    response = _specialist_result(
        ToolErrorAgent(),
        {
            "messages": [HumanMessage(content="How many albums are there?")],
            "current_request": "How many albums are there?",
        },
        {},
        invoice=False,
    )

    assert "I won't guess" in response


def test_invoice_agent_uses_four_customer_scoped_tools() -> None:
    from langchain_openai import ChatOpenAI

    agent = build_invoice_agent(
        ChatOpenAI(api_key="test-key", model="test-model")
    )

    assert agent.name == "invoice_agent"
    tools = build_invoice_tools()
    assert len(tools) == 4
    for tool in tools:
        assert "customer_id" not in tool.args


def test_invoice_tools_take_customer_id_only_from_trusted_config() -> None:
    tools = build_invoice_tools()
    invoice_list = tools[0]

    without_identity = json.loads(invoice_list.invoke({}))
    with_identity = json.loads(
        invoice_list.invoke(
            {},
            config={"configurable": {"verified_customer_id": "2"}},
        )
    )

    assert without_identity["code"] == "VERIFICATION_REQUIRED"
    assert isinstance(with_identity, list)
    assert with_identity
    assert all("invoice_id" in invoice for invoice in with_identity)


def test_invoice_tool_cannot_change_verified_customer_id() -> None:
    tool = build_invoice_tools()[2]

    result = json.loads(
        tool.invoke(
            {"invoice_id": "1"},
            config={"configurable": {"verified_customer_id": 2}},
        )
    )

    assert result["invoice_id"] == 1


def test_support_graph_has_verification_supervisor_and_memory_nodes() -> None:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key="test-key",
        model="test-model",
    )
    graph = build_support_graph(
        model=llm,
        verification_llm=EchoIdentifierLLM(),
        checkpointer=MemorySaver(),
        supervisor=RecordingSupervisor(),
        memory_model=EmptyPreferenceLLM(),
    )

    node_names = set(graph.get_graph().nodes)
    assert {
        "verify_info",
        "human_input",
        "load_memory",
        "supervisor",
        "create_memory",
    }.issubset(node_names)
    assert END in node_names


def test_supervisor_node_propagates_only_verified_identity_in_config() -> None:
    recording_supervisor = RecordingSupervisor()
    from src.agents.supervisor import _make_supervisor_node

    node = _make_supervisor_node(recording_supervisor)
    state = {
        "messages": [HumanMessage(content="Show my invoices")],
        "customer_id": 7,
        "verified": True,
    }

    result = node(state, {"configurable": {"thread_id": "agent-security-test"}})

    assert result["messages"] == state["messages"]
    assert recording_supervisor.config["configurable"]["verified_customer_id"] == 7
    assert recording_supervisor.config["configurable"]["thread_id"] == "agent-security-test"


def test_support_graph_verifies_then_routes_to_supervisor() -> None:
    supervisor = RecordingSupervisor()
    graph = build_support_graph(
        verification_llm=EchoIdentifierLLM(),
        checkpointer=MemorySaver(),
        supervisor=supervisor,
        memory_model=EmptyPreferenceLLM(),
    )
    config = {"configurable": {"thread_id": "support-flow-valid"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="5")]},
        config=config,
    )

    assert result["customer_id"] == 5
    assert result["verified"] is True
    assert supervisor.config["configurable"]["verified_customer_id"] == 5
    assert any(message.type == "system" for message in result["messages"])


def test_support_graph_interrupts_and_resumes_before_supervisor() -> None:
    supervisor = RecordingSupervisor()
    graph = build_support_graph(
        verification_llm=EchoIdentifierLLM(),
        checkpointer=MemorySaver(),
        supervisor=supervisor,
        memory_model=EmptyPreferenceLLM(),
    )
    config = {"configurable": {"thread_id": "support-flow-resume"}}

    paused = graph.invoke(
        {"messages": [HumanMessage(content="99999")]},
        config=config,
    )
    assert paused["verified"] is False
    assert supervisor.config is None

    result = graph.invoke(Command(resume="5"), config=config)

    assert result["verified"] is True
    assert result["customer_id"] == 5
    assert supervisor.config["configurable"]["verified_customer_id"] == 5


def test_support_graph_saves_preferences_and_loads_them_next_turn() -> None:
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    supervisor = RecordingSupervisor()
    graph = build_support_graph(
        verification_llm=EchoIdentifierLLM(),
        checkpointer=MemorySaver(),
        supervisor=supervisor,
        memory_model=JazzPreferenceLLM(),
        store=store,
    )
    config = {"configurable": {"thread_id": "support-memory-round-trip"}}

    graph.invoke(
        {"messages": [HumanMessage(content="5")]},
        config=config,
    )

    graph.invoke(
        {"messages": [HumanMessage(content="I love jazz")]},
        config=config,
    )

    saved = store.get((MEMORY_NAMESPACE, "5"), MEMORY_KEY)
    assert saved is not None
    assert saved.value["music_preferences"] == ["jazz"]

    graph.invoke(
        {"messages": [HumanMessage(content="Recommend something for me")]},
        config=config,
    )

    second_turn_messages = supervisor.inputs[-1]["messages"]
    preference_messages = [
        message
        for message in second_turn_messages
        if getattr(message, "id", None) == "saved-music-preferences"
    ]
    assert len(preference_messages) == 1
    assert "Music Preferences: jazz" in preference_messages[0].content
