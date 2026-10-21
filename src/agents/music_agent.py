from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.config import settings
from src.tools.music import (
    check_for_songs,
    get_albums_by_artist,
    get_songs_by_genre,
    get_track_details,
    get_tracks_by_artist,
)

from .prompts import MUSIC_AGENT_PROMPT

MUSIC_TOOLS = [
    get_albums_by_artist,
    get_tracks_by_artist,
    get_songs_by_genre,
    check_for_songs,
    get_track_details,
]


class MusicAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _configured_model(model=None):
    if model is not None:
        return model
    if not settings.api_key:
        raise ValueError("OPENAI_API_KEY must be configured for the music agent")
    if not settings.model_name:
        raise ValueError("OPENAI_MODEL must be configured for the music agent")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=settings.api_key,
        model=settings.model_name,
        temperature=settings.temperature,
        base_url=settings.api_base_url,
    )


def _route_after_assistant(state: MusicAgentState) -> str:
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END


def build_music_agent(model=None):
    llm_with_tools = _configured_model(model).bind_tools(MUSIC_TOOLS)

    def assistant(state: MusicAgentState) -> dict[str, list[BaseMessage]]:
        conversation_messages = state["messages"]
        saved_preferences = next(
            (
                message.content
                for message in reversed(conversation_messages)
                if isinstance(message, SystemMessage)
                and message.id == "saved-music-preferences"
            ),
            None,
        )
        system_prompt = MUSIC_AGENT_PROMPT
        if saved_preferences:
            system_prompt = f"{system_prompt}\n{saved_preferences}"
        response = llm_with_tools.invoke(
            [SystemMessage(content=system_prompt), *conversation_messages]
        )
        return {"messages": [response]}

    builder = StateGraph(MusicAgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(MUSIC_TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", _route_after_assistant)
    builder.add_edge("tools", "assistant")
    return builder.compile(name="music_agent")
