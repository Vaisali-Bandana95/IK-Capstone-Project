from contextvars import Context
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.ui.chat import (
    _user_facing_error,
    create_demo,
    reset_conversation,
    respond,
)


class FakeGraph:
    def __init__(self) -> None:
        self.snapshots: dict[str, Any] = {}
        self.inputs: list[tuple[Any, dict[str, Any]]] = []

    def get_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        return self.snapshots.get(
            thread_id,
            SimpleNamespace(next=(), tasks=(), values={}),
        )

    def stream(self, graph_input, config, stream_mode):
        thread_id = config["configurable"]["thread_id"]
        self.inputs.append((graph_input, config))
        if isinstance(graph_input, Command):
            self.snapshots[thread_id] = SimpleNamespace(
                next=(),
                tasks=(),
                values={"messages": [AIMessage(content="Verified and complete.")]},
            )
        elif graph_input["messages"][0].content == "need verification":
            pending = SimpleNamespace(
                interrupts=[
                    SimpleNamespace(
                        value="Please provide your Customer ID, email address, or phone number."
                    )
                ]
            )
            self.snapshots[thread_id] = SimpleNamespace(
                next=("human_input",),
                tasks=[pending],
                values={"messages": []},
            )
        else:
            self.snapshots[thread_id] = SimpleNamespace(
                next=(),
                tasks=(),
                values={"messages": [AIMessage(content="Here is the response.")]},
            )
        yield {"values": self.snapshots[thread_id].values}


def test_ui_shows_user_message_before_graph_completion() -> None:
    graph = FakeGraph()
    updates = respond("Hello", [], "thread-a", graph)

    first_update = next(updates)

    assert first_update[0] == [{"role": "user", "content": "Hello"}]
    assert "Processing" in first_update[1]
    assert first_update[2]["interactive"] is False
    assert first_update[4]["interactive"] is False
    assert graph.inputs == []

    final_update = list(updates)[-1]
    assert final_update[0][-1] == {
        "role": "assistant",
        "content": "Here is the response.",
    }
    assert "Complete in" in final_update[1]
    assert final_update[2]["interactive"] is True
    assert final_update[4]["interactive"] is True


def test_ui_interrupt_prompt_is_shown_and_next_message_resumes() -> None:
    graph = FakeGraph()
    updates = list(respond("need verification", [], "thread-b", graph))

    assert "Waiting for your input" in updates[-1][1]
    assert "Customer ID, email address, or phone number" in updates[-1][0][-1][
        "content"
    ]
    assert updates[-1][2]["interactive"] is True
    assert updates[-1][4]["interactive"] is True

    resumed = list(respond("5", updates[-1][0], "thread-b", graph))

    assert isinstance(graph.inputs[-1][0], Command)
    assert graph.inputs[-1][0].resume == "5"
    assert resumed[-1][0][-1]["content"] == "Verified and complete."


def test_ui_response_generator_can_resume_in_a_different_context() -> None:
    graph = FakeGraph()
    updates = respond("need verification", [], "context-switch-thread", graph)
    first_context = Context()
    second_context = Context()
    third_context = Context()

    first = first_context.run(next, updates)
    response = second_context.run(next, updates)
    with pytest.raises(StopIteration):
        third_context.run(next, updates)

    assert "Processing" in first[1]
    assert "Waiting for your input" in response[1]
    assert "Customer ID, email address, or phone number" in (
        response[0][-1]["content"]
    )


def test_each_chat_session_uses_its_own_thread_id() -> None:
    graph = FakeGraph()

    list(respond("Hello", [], "session-one", graph))
    list(respond("Hello", [], "session-two", graph))

    assert [
        config["configurable"]["thread_id"] for _, config in graph.inputs
    ] == ["session-one", "session-two"]


def test_ui_displays_actionable_timeout_message_without_exception_details() -> None:
    class TimedOutGraph(FakeGraph):
        def get_state(self, config):
            raise TimeoutError("internal provider details")

    updates = list(respond("Show my invoices", [], "timeout-thread", TimedOutGraph()))

    response = updates[-1][0][-1]["content"]
    assert "timed out" in response
    assert "try again in a moment" in response
    assert "internal provider details" not in response


def test_ui_displays_safe_recovery_steps_for_unknown_errors() -> None:
    class BrokenGraph(FakeGraph):
        def get_state(self, config):
            raise RuntimeError("internal stack trace")

    updates = list(respond("Show my invoices", [], "error-thread", BrokenGraph()))

    response = updates[-1][0][-1]["content"]
    assert "start a new conversation or contact support" in response
    assert "internal stack trace" not in response


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ConnectionError("private connection details"), "could not be reached"),
        (
            type("RateLimitError", (Exception,), {})("private quota details"),
            "service is busy",
        ),
    ],
)
def test_ui_error_categories_give_safe_recovery_guidance(
    error: Exception,
    expected: str,
) -> None:
    message = _user_facing_error(error)

    assert expected in message
    assert "private" not in message


def test_reset_clears_chat_and_creates_a_new_thread_id() -> None:
    history, status, message, thread_id, send_update = reset_conversation()

    assert history == []
    assert "New conversation" in status
    assert message == ""
    assert thread_id
    assert status
    assert send_update["interactive"] is True
    assert thread_id != reset_conversation()[3]


def test_create_demo_builds_gradio_blocks_without_live_model() -> None:
    import gradio as gr

    demo = create_demo(graph=FakeGraph())

    assert isinstance(demo, gr.Blocks)
    config = demo.get_config_file()
    button_values = [
        component.get("props", {}).get("value", "")
        for component in config["components"]
        if component.get("type") == "button"
    ]
    assert not any("&nbsp;" in value for value in button_values)
