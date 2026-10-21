from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Checkpointer
from langgraph_supervisor import create_supervisor

from src.config import settings
from src.graph.verification import (
    VerificationState,
    _exhausted_node,
    _human_input_node,
    create_verification_node,
    route_after_verification,
)
from src.graph.memory import (
    create_memory_node,
    create_memory_store,
    load_memory,
)

from .invoice_agent import build_invoice_agent
from .music_agent import build_music_agent
from .prompts import SUPERVISOR_PROMPT


class SupportState(VerificationState, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    preferences: dict[str, Any]
    music_preferences: list[str]
    music_preferences_context: str


def _configured_model(model=None):
    if model is not None:
        return model
    if not settings.api_key:
        raise ValueError("OPENAI_API_KEY must be configured for the supervisor")
    if not settings.model_name:
        raise ValueError("OPENAI_MODEL must be configured for the supervisor")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=settings.api_key,
        model=settings.model_name,
        temperature=settings.temperature,
        base_url=settings.api_base_url,
    )


def build_supervisor(model=None, music_agent=None, invoice_agent=None):
    llm = _configured_model(model)
    music_agent = music_agent or build_music_agent(llm)
    invoice_agent = invoice_agent or build_invoice_agent(llm)
    return create_supervisor(
        agents=[invoice_agent, music_agent],
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        supervisor_name="support_supervisor",
    ).compile()


def _make_supervisor_node(supervisor):
    def supervisor_node(
        state: SupportState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        if not state.get("verified") or state.get("customer_id") is None:
            raise ValueError("Supervisor cannot run before customer verification")

        configurable = {
            **config.get("configurable", {}),
            "verified_customer_id": state["customer_id"],
        }
        child_config = {**config, "configurable": configurable}
        result = supervisor.invoke(
            {"messages": state["messages"]},
            config=child_config,
        )
        return {"messages": result["messages"]}

    return supervisor_node


def build_support_graph(
    model=None,
    verification_llm=None,
    checkpointer: Checkpointer | None = None,
    supervisor=None,
    store: BaseStore | None = None,
    memory_model=None,
):
    supervisor_graph = supervisor or build_supervisor(model)
    memory_store = store if store is not None else create_memory_store()
    builder = StateGraph(SupportState)
    builder.add_node("verify_info", create_verification_node(verification_llm))
    builder.add_node("human_input", _human_input_node)
    builder.add_node("verification_exhausted", _exhausted_node)
    builder.add_node("load_memory", load_memory)
    builder.add_node("supervisor", _make_supervisor_node(supervisor_graph))
    builder.add_node("create_memory", create_memory_node(memory_model))

    builder.add_edge(START, "verify_info")
    builder.add_conditional_edges(
        "verify_info",
        route_after_verification,
        {
            "next": "load_memory",
            "human_input": "human_input",
            "exhausted": "verification_exhausted",
        },
    )
    builder.add_edge("human_input", "verify_info")
    builder.add_edge("load_memory", "supervisor")
    builder.add_edge("supervisor", "create_memory")
    builder.add_edge("create_memory", END)
    builder.add_edge("verification_exhausted", END)
    return builder.compile(
        checkpointer=checkpointer or MemorySaver(),
        store=memory_store,
    )
