from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.graph.persistence import create_persistent_checkpointer


def _build_graph(checkpointer):
    builder = StateGraph(MessagesState)
    builder.add_node("retain_state", lambda state: {})
    builder.add_edge(START, "retain_state")
    builder.add_edge("retain_state", END)
    return builder.compile(checkpointer=checkpointer)


def test_conversation_checkpoint_survives_database_reopen(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "conversation.sqlite3"
    config = {"configurable": {"thread_id": "persistent-thread"}}
    first_saver = create_persistent_checkpointer(checkpoint_path)

    _build_graph(first_saver).invoke(
        {"messages": [HumanMessage(content="Keep this conversation")]},
        config=config,
    )
    first_saver.conn.close()

    second_saver = create_persistent_checkpointer(checkpoint_path)
    restored_state = _build_graph(second_saver).get_state(config)

    assert restored_state.values["messages"][-1].content == "Keep this conversation"
    second_saver.conn.close()
