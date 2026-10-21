import logging
import time
import uuid
from collections.abc import Iterator
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agents.supervisor import build_support_graph
from src.config import settings
from src.db import initialize_database

logger = logging.getLogger(__name__)


def _new_thread_id() -> str:
    return str(uuid.uuid4())


def _is_waiting_for_input(snapshot: Any) -> bool:
    next_nodes = set(getattr(snapshot, "next", ()) or ())
    if "human_input" in next_nodes:
        return True
    return any(
        getattr(task, "interrupts", ())
        for task in getattr(snapshot, "tasks", ())
    )


def _interrupt_prompt(snapshot: Any) -> str:
    for task in getattr(snapshot, "tasks", ()):
        for pending in getattr(task, "interrupts", ()):
            value = getattr(pending, "value", None)
            if isinstance(value, str) and value:
                return value
    return "Please provide your Customer ID, email address, or phone number."


def _latest_assistant_response(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        if (
            isinstance(message, AIMessage)
            and not getattr(message, "tool_calls", None)
            and isinstance(message.content, str)
            and message.content.strip()
        ):
            return message.content.strip()
    return ""


def respond(
    message: str,
    history: list[dict[str, str]] | None,
    thread_id: str | None,
    graph: Any,
) -> Iterator[tuple[list[dict[str, str]], str, Any, str, Any]]:
    chat_history = list(history or [])
    session_id = thread_id or _new_thread_id()
    user_text = message.strip()
    if not user_text:
        yield (
            chat_history,
            "Enter a message to continue.",
            gr.update(interactive=True),
            session_id,
            gr.update(interactive=True),
        )
        return

    chat_history.append({"role": "user", "content": user_text})
    yield (
        chat_history,
        "⏳ Processing your message…",
        gr.update(value="", interactive=False),
        session_id,
        gr.update(interactive=False),
    )
    start_time = time.perf_counter()
    config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = graph.get_state(config)
        if _is_waiting_for_input(snapshot):
            graph_input = Command(resume=user_text)
        else:
            graph_input = {"messages": [HumanMessage(content=user_text)]}

        for _event in graph.stream(
            graph_input,
            config=config,
            stream_mode="values",
        ):
            pass

        snapshot = graph.get_state(config)
        if _is_waiting_for_input(snapshot):
            prompt = _interrupt_prompt(snapshot)
            chat_history.append({"role": "assistant", "content": prompt})
            yield (
                chat_history,
                "⏸️ Waiting for your input",
                gr.update(interactive=True),
                session_id,
                gr.update(interactive=True),
            )
            return

        response_text = _latest_assistant_response(snapshot.values or {})
        if response_text:
            chat_history.append({"role": "assistant", "content": response_text})
        elapsed = time.perf_counter() - start_time
        yield (
            chat_history,
            f"✅ Complete in {elapsed:.2f}s",
            gr.update(interactive=True),
            session_id,
            gr.update(interactive=True),
        )
    except Exception as error:
        logger.error("Chat graph invocation failed (%s)", type(error).__name__)
        chat_history.append(
            {
                "role": "assistant",
                "content": "Sorry, something went wrong. Please try again.",
            }
        )
        yield (
            chat_history,
            "❌ Error — please try again.",
            gr.update(interactive=True),
            session_id,
            gr.update(interactive=True),
        )


def reset_conversation() -> tuple[list[dict[str, str]], str, str, str, Any]:
    return (
        [],
        "🟢 New conversation. Please verify your account to continue.",
        "",
        _new_thread_id(),
        gr.update(interactive=True),
    )


def create_demo(graph: Any | None = None) -> gr.Blocks:
    initialize_database()
    support_graph = graph if graph is not None else build_support_graph()

    with gr.Blocks(title="Chinook Music Support") as demo:
        gr.Markdown(
            "# Chinook Music Support\n"
            "Ask about the music catalog or your invoices. "
            "Verify with your Customer ID, email, or phone when prompted."
        )
        thread_state = gr.State(value=_new_thread_id)
        chatbot = gr.Chatbot(
            type="messages",
            label="Conversation",
            height=520,
            allow_tags=False,
        )
        status = gr.Markdown("🟢 Ready")
        with gr.Row():
            message_input = gr.Textbox(
                label="Message",
                placeholder="Ask about albums, tracks, or invoices…",
                scale=8,
                lines=1,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)
            reset_button = gr.Button("New conversation", scale=1)

        def submit(message, history, session_id):
            yield from respond(message, history, session_id, support_graph)

        send_button.click(
            submit,
            inputs=[message_input, chatbot, thread_state],
            outputs=[chatbot, status, message_input, thread_state, send_button],
            api_name="chat",
        )
        message_input.submit(
            submit,
            inputs=[message_input, chatbot, thread_state],
            outputs=[chatbot, status, message_input, thread_state, send_button],
            api_name="chat_submit",
        )
        reset_button.click(
            reset_conversation,
            inputs=[],
            outputs=[chatbot, status, message_input, thread_state, send_button],
            api_name="reset",
        )

    return demo
