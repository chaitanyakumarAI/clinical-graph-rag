"""pipeline/query_planner.py — route parsing, validation and fallbacks."""

import pytest

from llm import RateLimitExhausted
from pipeline import query_planner
from pipeline.query_planner import QueryPlan, plan_query


def _plan(**overrides):
    base = {
        "route": "hybrid_single",
        "reason": "one hop",
        "entities_mentioned": ["Metformin"],
        "confidence": 0.9,
        "retrieval_strategy": "single-hop Cypher",
        "target_relations": ["CAUSES"],
        "target_labels": ["SideEffect"],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("route", ["vector_only", "hybrid_single", "hybrid_multi", "gds_augmented"])
def test_each_route_is_accepted(monkeypatch, fake_llm, route):
    monkeypatch.setattr(query_planner, "chat_json", fake_llm(_plan(route=route)))
    assert plan_query("q").route == route


@pytest.mark.parametrize("legacy, expected", [
    ("no_graph", "vector_only"),
    ("single_hop", "hybrid_single"),
    ("multi_hop", "hybrid_multi"),
])
def test_legacy_route_names_are_mapped(monkeypatch, fake_llm, legacy, expected):
    monkeypatch.setattr(query_planner, "chat_json", fake_llm(_plan(route=legacy)))
    assert plan_query("q").route == expected


def test_relations_are_whitelisted_and_uppercased(monkeypatch, fake_llm):
    monkeypatch.setattr(query_planner, "chat_json", fake_llm(_plan(
        target_relations=["causes", "DROP_ALL", "TREATS]->() DETACH DELETE n //", 7, "interacts_with"],
    )))

    plan = plan_query("q")

    # Anything not in the 16 known relation types is dropped, so nothing an LLM
    # writes here can reach the f-string Cypher in graph_retriever.
    assert plan.target_relations == ["CAUSES", "INTERACTS_WITH"]


def test_unknown_route_falls_back_to_hybrid_multi(monkeypatch, fake_llm):
    monkeypatch.setattr(query_planner, "chat_json", fake_llm(_plan(route="teleport")))

    plan = plan_query("q")

    assert plan.route == "hybrid_multi"
    assert plan.confidence == pytest.approx(0.3)
    assert plan.entities_mentioned == []


def test_malformed_reply_falls_back(monkeypatch, fake_llm):
    monkeypatch.setattr(query_planner, "chat_json", fake_llm(ValueError("Expecting value")))
    assert plan_query("q").route == "hybrid_multi"


def test_rate_limit_propagates(monkeypatch, fake_llm):
    monkeypatch.setattr(query_planner, "chat_json", fake_llm(RateLimitExhausted("groq", None, "x")))
    with pytest.raises(RateLimitExhausted):
        plan_query("q")


@pytest.mark.parametrize("raw, expected", [(1.7, 1.0), (-0.2, 0.0), ("0.65", 0.65)])
def test_confidence_is_clamped(raw, expected):
    plan = QueryPlan(route="vector_only", reason="r", confidence=raw)
    assert plan.confidence == pytest.approx(expected)


def test_context_chunks_are_truncated_into_prompt(monkeypatch, fake_llm):
    llm = fake_llm(_plan())
    monkeypatch.setattr(query_planner, "chat_json", llm)
    chunks = [f"chunk{i} " + "x" * 500 for i in range(6)]

    plan_query("Which drugs treat T2DM?", context_chunks=chunks)

    prompt = llm.calls[0]["user"]
    assert prompt.startswith("Question: Which drugs treat T2DM?")
    # only the first 4 chunks, 300 chars each
    assert "chunk3" in prompt and "chunk4" not in prompt
    assert "x" * 301 not in prompt
