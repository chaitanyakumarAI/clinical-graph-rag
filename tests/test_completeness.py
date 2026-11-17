"""pipeline/completeness.py and the required_parts option of the generator."""

import pytest

from llm import RateLimitExhausted
from pipeline import completeness, generator
from pipeline.completeness import CoverageReport, check_coverage, should_check


def test_report_lists_missing_parts():
    report = CoverageReport(parts=[
        {"part": "which drugs interact", "covered": True},
        {"part": "what the effect is", "covered": False},
    ])
    assert report.part_names == ["which drugs interact", "what the effect is"]
    assert report.missing == ["what the effect is"]


def test_report_drops_malformed_items_and_caps_parts():
    report = CoverageReport(parts=[
        {"part": "  a  ", "covered": True},
        {"part": "", "covered": False},
        {"covered": False},
        "just a string",
        {"part": "b", "covered": "yes"},      # only a real boolean true counts as covered
        {"part": "c"}, {"part": "d", "covered": True}, {"part": "e", "covered": True},
    ])
    assert report.part_names == ["a", "b", "c", "d"]          # MAX_PARTS = 4
    assert report.missing == ["b", "c"]
    assert CoverageReport(parts=None).parts == []


@pytest.mark.parametrize("route, sub_queries, expected", [
    ("vector_only", [], False),           # definitions: nothing to drop
    ("vector_only", ["a", "b"], True),    # rewriter found several parts
    ("hybrid_single", [], True),          # planner often labels multi-part questions this way
    ("hybrid_multi", [], True),
    ("gds_augmented", [], True),
])
def test_should_check(route, sub_queries, expected):
    assert should_check(route, sub_queries) is expected


def test_check_coverage_sends_question_sub_questions_and_answer(monkeypatch, fake_llm):
    llm = fake_llm({"parts": [{"part": "x", "covered": False}]})
    monkeypatch.setattr(completeness, "chat_json", llm)

    report = check_coverage("Q?", "the draft", sub_queries=["sub one", "sub two"])

    assert report.missing == ["x"]
    prompt = llm.calls[0]["user"]
    assert prompt.startswith("QUESTION: Q?")
    assert "- sub one\n- sub two" in prompt
    assert prompt.endswith("ANSWER:\nthe draft")


def test_check_coverage_fails_open(monkeypatch, fake_llm):
    monkeypatch.setattr(completeness, "chat_json", fake_llm(ValueError("not JSON")))
    assert check_coverage("Q?", "draft").missing == []


def test_check_coverage_propagates_rate_limit(monkeypatch, fake_llm):
    monkeypatch.setattr(completeness, "chat_json", fake_llm(RateLimitExhausted("groq", 60, "x")))
    with pytest.raises(RateLimitExhausted):
        check_coverage("Q?", "draft")


def test_generator_lists_required_parts_before_the_answer_cue(monkeypatch, fake_llm):
    llm = fake_llm("answer")
    monkeypatch.setattr(generator, "chat", llm)

    generator.generate_answer("Q?", "CTX", required_parts=["first part", "second part"])

    prompt = llm.calls[0]["user"]
    assert "1. first part\n2. second part" in prompt
    assert prompt.index("second part") < prompt.index("Answer (with inline citations):")
    assert prompt.rstrip().endswith("Answer (with inline citations):")


def test_generator_prompt_unchanged_without_parts(monkeypatch, fake_llm):
    llm = fake_llm("answer")
    monkeypatch.setattr(generator, "chat", llm)
    generator.generate_answer("Q?", "CTX")
    assert llm.calls[0]["user"] == generator.USER_TEMPLATE.format(context="CTX", query="Q?")
