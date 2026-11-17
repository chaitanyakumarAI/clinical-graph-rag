"""pipeline/query_rewriter.py — rewriting, sub-query handling and fallbacks."""

import pytest

from llm import RateLimitExhausted
from pipeline import query_rewriter
from pipeline.query_rewriter import RewrittenQuery, no_rewrite, rewrite_query


def test_rewrite_uses_llm_output(monkeypatch, fake_llm):
    llm = fake_llm({"rewritten": "What are the side effects of metformin?", "sub_queries": []})
    monkeypatch.setattr(query_rewriter, "chat_json", llm)

    result = rewrite_query("side efects of metformn?")

    assert result.original == "side efects of metformn?"
    assert result.rewritten == "What are the side effects of metformin?"
    assert result.sub_queries == []
    assert result.search_queries == ["What are the side effects of metformin?"]
    assert llm.calls[0]["user"] == "Q: side efects of metformn?"


def test_sub_queries_are_cleaned_and_capped(monkeypatch, fake_llm):
    monkeypatch.setattr(query_rewriter, "chat_json", fake_llm({
        "rewritten": "Which diabetes drugs interact with blood pressure drugs?",
        "sub_queries": ["  diabetes drugs  ", "", 42, "blood pressure drugs", "third", "fourth"],
    }))

    result = rewrite_query("which T2DM drugs interact with BP meds")

    # blanks and non-strings dropped, whitespace stripped, at most MAX_SUB_QUERIES kept
    assert result.sub_queries == ["diabetes drugs", "blood pressure drugs", "third"]
    assert len(result.sub_queries) == query_rewriter.MAX_SUB_QUERIES


def test_search_queries_deduplicate_case_insensitively():
    query = RewrittenQuery(
        original="q",
        rewritten="Metformin side effects",
        sub_queries=["metformin SIDE effects", "lactic acidosis risk"],
    )
    assert query.search_queries == ["Metformin side effects", "lactic acidosis risk"]


def test_empty_rewrite_falls_back_to_original(monkeypatch, fake_llm):
    monkeypatch.setattr(query_rewriter, "chat_json", fake_llm({"rewritten": "   ", "sub_queries": None}))

    result = rewrite_query("AF symptoms")

    assert result.rewritten == "AF symptoms"
    assert result.sub_queries == []


@pytest.mark.parametrize("error", [ValueError("not JSON"), KeyError("x"), RuntimeError("boom")])
def test_llm_failure_returns_pass_through(monkeypatch, fake_llm, error):
    monkeypatch.setattr(query_rewriter, "chat_json", fake_llm(error))

    result = rewrite_query("why is high BP called silent")

    assert result == no_rewrite("why is high BP called silent")


def test_rate_limit_is_not_swallowed(monkeypatch, fake_llm):
    monkeypatch.setattr(query_rewriter, "chat_json",
                        fake_llm(RateLimitExhausted("groq", 3600, "daily quota")))
    with pytest.raises(RateLimitExhausted):
        rewrite_query("anything")


def test_no_rewrite_is_identity():
    result = no_rewrite("Exact question?")
    assert result.rewritten == result.original == "Exact question?"
    assert result.search_queries == ["Exact question?"]
