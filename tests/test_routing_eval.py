import pytest
from langchain_core.messages import HumanMessage

from src.agents.supervisor import (
    RouteDecision,
    _make_route_node,
    _obvious_route,
)


@pytest.mark.parametrize(
    ("utterance", "expected_route"),
    [
        ("Find tracks from Norah Jones", "music"),
        ("Albums by AC/DC, please", "music"),
        ("Could you suggest jazz?", "music"),
        ("I really like jazz", "music"),
        ("What genre is the song I bought?", "music"),
        ("What is the release date for this album?", "music"),
        ("Show my invoices", "invoice"),
        ("How much did I spend?", "invoice"),
        ("What songs have I purchased?", "invoice"),
        ("Can I get a receipt?", "invoice"),
        ("Show my invoices and recommend jazz albums", "mixed"),
        ("List my purchases and find rock tracks", "mixed"),
        ("Which songs did I buy, and what genre is AC/DC?", "mixed"),
        ("Show my invoices and the genre of a track", "mixed"),
        ("What's the weather today?", "off_topic"),
        ("Give me the stock market forecast", "off_topic"),
        ("Who won the football scores?", "off_topic"),
        ("What are the election results?", "off_topic"),
        ("Customer ID is 5", "verify"),
        ("My email is customer@example.com", "verify"),
    ],
)
def test_routing_evaluation_phrases(utterance: str, expected_route: str) -> None:
    assert _obvious_route(utterance) == expected_route


class OffTopicFallbackModel:
    def __init__(self) -> None:
        self.calls = 0

    def with_structured_output(self, schema):
        assert schema is RouteDecision
        return self

    def invoke(self, messages):
        self.calls += 1
        assert messages[-1].content == "Can you explain how clouds form?"
        return RouteDecision(intent="off_topic")


def test_unrecognized_request_is_delegated_to_route_classifier() -> None:
    model = OffTopicFallbackModel()
    route_node = _make_route_node(model)

    result = route_node(
        {"messages": [HumanMessage(content="Can you explain how clouds form?")]},
        {},
    )

    assert model.calls == 1
    assert result["route"] == "off_topic"
