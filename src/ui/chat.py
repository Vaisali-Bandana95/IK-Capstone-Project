import logging
import time
import uuid
from collections.abc import Iterator
from typing import Any

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.agents.supervisor import build_support_graph
from src.config.logging import (
    log_event,
    reset_correlation_id,
    set_correlation_id,
)
from src.db import initialize_database
from src.graph.persistence import create_persistent_checkpointer

logger = logging.getLogger(__name__)

APP_CSS = """
:root {
    --ink: #172b4d;
    --muted: #62748b;
    --line: #e2e8f0;
    --brand: #2563eb;
}

body, .gradio-container {
    background: #f3f6fb !important;
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
}

#app-shell {
    box-sizing: border-box;
    width: min(1440px, 100%);
    margin: 0 auto;
    padding: 30px 34px 36px;
    gap: 20px;
}

#brand-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 25px 30px;
    border: 1px solid #203f64;
    border-radius: 18px;
    background: linear-gradient(115deg, #10243d 0%, #173b63 70%, #1c4d78 100%);
    color: #fff;
    box-shadow: 0 12px 28px rgba(20, 45, 76, .12);
}

#brand-header .brand-lockup {
    display: flex;
    align-items: center;
    gap: 16px;
}

#brand-header .brand-mark {
    display: grid;
    width: 48px;
    height: 48px;
    place-items: center;
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 14px;
    background: rgba(255,255,255,.1);
    color: #bfdbfe;
    font-size: 24px;
}

#brand-header h1 {
    margin: 0;
    color: #fff;
    font-size: clamp(22px, 3vw, 29px);
    font-weight: 700;
    letter-spacing: -.035em;
}

#brand-header p {
    margin: 5px 0 0;
    color: #c9d8e9;
    font-size: 14px;
}

#brand-header .secure-label {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    flex-shrink: 0;
    padding: 9px 13px;
    border: 1px solid rgba(191,219,254,.25);
    border-radius: 999px;
    background: rgba(255,255,255,.08);
    color: #e4efff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

#workspace {
    align-items: stretch;
    gap: 20px;
}

#support-sidebar, #conversation-panel {
    box-sizing: border-box;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: #fff;
    box-shadow: 0 8px 24px rgba(27, 51, 84, .045);
}

#support-sidebar {
    gap: 16px;
    padding: 23px 20px;
}

#support-sidebar .eyebrow {
    margin: 0 0 7px;
    color: #647995;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: .12em;
    text-transform: uppercase;
}

#support-sidebar h2 {
    margin: 0;
    color: var(--ink);
    font-size: 20px;
    letter-spacing: -.025em;
}

#support-sidebar .sidebar-copy {
    margin: 7px 0 0;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
}

#support-sidebar .section-label {
    margin: 10px 0 0;
    color: #536780;
    font-size: 12px;
    font-weight: 700;
}

#support-sidebar .prompt-button button {
    justify-content: flex-start;
    min-height: 44px;
    border: 1px solid #e4eaf2;
    border-radius: 10px;
    background: #fff;
    color: #263d5a;
    font-size: 13px;
    font-weight: 600;
    transition: border-color .16s ease, background .16s ease, transform .16s ease;
}

#support-sidebar .prompt-button button:hover {
    transform: translateY(-1px);
    border-color: #b7cdf4;
    background: #f5f8ff;
}

#support-sidebar .privacy-card {
    padding: 14px;
    border: 1px solid #dce9fb;
    border-radius: 12px;
    background: #f5f9ff;
}

#support-sidebar .privacy-card strong {
    display: block;
    margin-bottom: 5px;
    color: #244b7a;
    font-size: 12px;
}

#support-sidebar .privacy-card span {
    display: block;
    color: #5e7088;
    font-size: 11px;
    line-height: 1.5;
}

#conversation-panel {
    min-width: 0;
    gap: 14px;
    padding: 20px;
}

#conversation-heading {
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 2px 2px 12px;
    border-bottom: 1px solid #edf1f6;
}

#conversation-heading h2 {
    margin: 0;
    color: var(--ink);
    font-size: 17px;
    font-weight: 700;
}

#conversation-heading p {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 12px;
}

#status-banner {
    flex: 0 0 auto !important;
    width: auto;
    min-width: max-content;
    padding: 7px 11px;
    border: 1px solid #dbeafe;
    border-radius: 999px;
    background: #eff6ff;
    color: #28518b;
    font-size: 11px;
    font-weight: 650;
    white-space: nowrap;
}

#status-banner p {
    margin: 0;
}

#chatbot, #chatbot .wrap, #chatbot .bubble-wrap {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#chatbot {
    min-height: 460px;
}

#chatbot .message {
    border-radius: 14px !important;
    box-shadow: none !important;
}

#chatbot .message.user {
    border: 1px solid #cfe0ff !important;
    background: #edf4ff !important;
}

#chatbot .message.bot {
    border: 1px solid #e6ebf2 !important;
    background: #fff !important;
}

#composer {
    align-items: center;
    gap: 10px;
    padding-top: 13px;
    border-top: 1px solid #edf1f6;
}

#message-input textarea {
    min-height: 48px !important;
    border: 1px solid #dce4ee !important;
    border-radius: 12px !important;
    background: #fbfcfe !important;
    box-shadow: none !important;
}

#message-input textarea:focus {
    border-color: #8db4f5 !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.1) !important;
}

#send-button button {
    min-height: 46px;
    padding: 0 22px;
    border-radius: 11px;
    background: #2563eb;
    font-weight: 650;
    box-shadow: 0 5px 12px rgba(37,99,235,.18);
}

#reset-button button {
    min-height: 44px;
    min-width: 138px;
    border: 1px solid #e1e7ef;
    border-radius: 10px;
    background: #fff;
    color: #42566f;
    font-size: 12px;
    font-weight: 600;
}

#workspace .column {
    min-width: 0;
}

footer {
    display: none !important;
}

@media (max-width: 860px) {
    #app-shell {
        padding: 18px 14px 24px;
    }

    #workspace {
        flex-direction: column !important;
    }

    #brand-header {
        align-items: flex-start;
        padding: 21px;
    }

    #brand-header .secure-label {
        display: none;
    }

    #support-sidebar {
        padding: 19px;
    }

    #chatbot {
        min-height: 360px;
    }
}

@media (max-width: 520px) {
    #conversation-panel {
        padding: 14px;
    }

    #conversation-heading {
        align-items: flex-start;
        flex-direction: column;
    }

    #composer {
        flex-wrap: wrap;
    }

    #message-input {
        flex-basis: 100%;
    }
}
"""


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
    correlation_token = set_correlation_id(session_id)
    config = {"configurable": {"thread_id": session_id}}
    log_event(logger, "chat.request_started")

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
            log_event(
                logger,
                "chat.verification_interrupted",
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
                outcome="waiting_for_input",
            )
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
        log_event(
            logger,
            "chat.request_completed",
            duration_ms=round(elapsed * 1000, 2),
            outcome="success",
        )
        yield (
            chat_history,
            f"✅ Complete in {elapsed:.2f}s",
            gr.update(interactive=True),
            session_id,
            gr.update(interactive=True),
        )
    except Exception as error:  # noqa: BLE001 - shield UI from graph/library failures
        log_event(
            logger,
            "chat.request_failed",
            level=logging.ERROR,
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            error_type=type(error).__name__,
            outcome="error",
        )
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
    finally:
        reset_correlation_id(correlation_token)


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
    if graph is None:
        support_graph = build_support_graph(
            checkpointer=create_persistent_checkpointer()
        )
    else:
        support_graph = graph

    with gr.Blocks(
        title="Chinook Music Support",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            radius_size="lg",
        ),
        css=APP_CSS,
    ) as demo:
        thread_state = gr.State(value=_new_thread_id)

        with gr.Column(elem_id="app-shell"):
            gr.HTML(
                """
                <header class="brand-lockup">
                    <div class="brand-lockup">
                        <div class="brand-mark" aria-hidden="true">♫</div>
                        <div>
                            <h1>Chinook Music Support</h1>
                            <p>Your personal concierge for music and account support.</p>
                        </div>
                    </div>
                    <div class="secure-label">
                        <span aria-hidden="true">●</span>
                        <span>Capstone demo portal</span>
                    </div>
                </header>
                """,
                elem_id="brand-header",
                padding=False,
            )

            with gr.Row(elem_id="workspace"):
                with gr.Column(
                    scale=1,
                    min_width=245,
                    elem_id="support-sidebar",
                ):
                    gr.HTML(
                        """
                        <div>
                            <p class="eyebrow">Support center</p>
                            <h2>How can we help?</h2>
                            <p class="sidebar-copy">
                                Explore the catalog or get help with your account.
                                Choose a starting point or ask in your own words.
                            </p>
                        </div>
                        """,
                        padding=False,
                    )
                    gr.Markdown("Popular topics", elem_classes=["section-label"])
                    catalog_prompt = gr.Button(
                        "♫ Explore the music catalog",
                        elem_classes=["prompt-button"],
                    )
                    invoice_prompt = gr.Button(
                        "▤ View my invoices",
                        elem_classes=["prompt-button"],
                    )
                    preferences_prompt = gr.Button(
                        "✧ Get music recommendations",
                        elem_classes=["prompt-button"],
                    )
                    gr.HTML(
                        """
                        <div class="privacy-card">
                            <strong>Protected account access</strong>
                            <span>
                                Demo account lookup is required before invoices or
                                purchases. Use your sample Customer ID, email, or
                                phone when prompted. Not real-world authentication.
                            </span>
                        </div>
                        """,
                        padding=False,
                    )

                with gr.Column(
                    scale=3,
                    min_width=480,
                    elem_id="conversation-panel",
                ):
                    with gr.Row(elem_id="conversation-heading"):
                        gr.HTML(
                            """
                            <div>
                                <h2>Conversation</h2>
                                <p>Ask a question and your support assistant will take it from here.</p>
                            </div>
                            """,
                            padding=False,
                        )
                        status = gr.Markdown("● Ready", elem_id="status-banner")

                    chatbot = gr.Chatbot(
                        type="messages",
                        show_label=False,
                        height=480,
                        placeholder=(
                            "Welcome to Chinook Music Support\n\n"
                            "Ask about artists, albums, genres, or your invoices. "
                            "We’ll guide you through account verification when needed."
                        ),
                        allow_tags=False,
                        elem_id="chatbot",
                    )

                    with gr.Row(elem_id="composer"):
                        message_input = gr.Textbox(
                            label="Message",
                            placeholder="Ask about music, albums, or your account…",
                            scale=8,
                            lines=1,
                            show_label=False,
                            container=False,
                            elem_id="message-input",
                        )
                        send_button = gr.Button(
                            "Send",
                            variant="primary",
                            scale=1,
                            elem_id="send-button",
                        )
                        reset_button = gr.Button(
                            "New conversation",
                            scale=1,
                            elem_id="reset-button",
                        )

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

        catalog_prompt.click(
            lambda: "What music do you have in the catalog?",
            outputs=message_input,
            queue=False,
        )
        invoice_prompt.click(
            lambda: "Show me my invoices",
            outputs=message_input,
            queue=False,
        )
        preferences_prompt.click(
            lambda: "Recommend music based on my preferences",
            outputs=message_input,
            queue=False,
        )

    return demo
