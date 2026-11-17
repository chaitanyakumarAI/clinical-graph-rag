"""
pipeline/completeness.py
Check that a draft answer covers every part of the question, and repair it.

Why:
  The RAGAS evaluation showed context recall ≈ 1.0 on every multi-hop question
  — the facts were retrieved — yet answer correctness was the lowest of any
  category, because the answer silently dropped one part of a multi-part
  question (e.g. listed the diabetes drugs but not the interaction).

How (one extra LLM call when everything is covered, two when it is not):
  1. check_coverage()  — the LLM splits the question into the parts it asks
     for (reusing the rewriter's sub-queries when there are any) and marks each
     part covered / missing in the draft answer.
  2. For each missing part, the caller runs a targeted vector search, adds the
     best new chunks to the context, and regenerates the answer once with the
     parts listed explicitly (see generator.generate_answer(required_parts=...)).
  No loop: at most one repair round, so the cost is bounded.
"""

from __future__ import annotations
from typing import List

from pydantic import BaseModel, field_validator

from llm import RateLimitExhausted, chat_json

MAX_PARTS         = 4   # parts the checker may list
MAX_REPAIR_PARTS  = 2   # missing parts we re-retrieve for

SYSTEM_PROMPT = """You check whether an answer fully addresses a medical question.

1. Split the QUESTION into the distinct things it asks for ("parts"). A simple
   question has one part. "Which X interact with Y, and what happens?" has two:
   which X interact with Y; what the effect of the interaction is.
   If SUB-QUESTIONS are given, use them as the parts.
2. For each part, decide if the ANSWER addresses it with specific information.
   A part is NOT covered if the answer skips it, only hints at it, or says the
   sources lack that information.

Judge coverage only — not whether the answer is medically correct.

Return ONLY JSON:
{"parts": [{"part": "short self-contained search query for this part", "covered": true}]}"""


class PartCoverage(BaseModel):
    part:    str
    covered: bool


class CoverageReport(BaseModel):
    parts: List[PartCoverage] = []

    @field_validator("parts", mode="before")
    @classmethod
    def clean_parts(cls, v):
        if not isinstance(v, list):
            return []
        cleaned = []
        for item in v:
            if isinstance(item, dict) and isinstance(item.get("part"), str) and item["part"].strip():
                cleaned.append({"part": item["part"].strip(),
                                "covered": item.get("covered") is True})
        return cleaned[:MAX_PARTS]

    @property
    def missing(self) -> List[str]:
        return [p.part for p in self.parts if not p.covered]

    @property
    def part_names(self) -> List[str]:
        return [p.part for p in self.parts]


def should_check(route: str, sub_queries: List[str]) -> bool:
    """
    Skip the check where it cannot help: a single-part vector_only question
    (definitions, mechanisms) has nothing to drop. Everything else is checked —
    the planner labels many multi-part questions hybrid_single, and the
    rewriter only splits a minority of them, so neither signal alone is enough.
    """
    return bool(sub_queries) or route != "vector_only"


def check_coverage(
    question:    str,
    answer:      str,
    sub_queries: List[str] = (),
    provider:    str = "auto",
) -> CoverageReport:
    """
    Ask the LLM which parts of the question the answer covers.

    Fails open: if the reply is unusable, returns an empty report (no parts →
    nothing missing), so a checker problem never breaks or delays the answer.
    """
    user = f"QUESTION: {question}\n"
    if sub_queries:
        user += "SUB-QUESTIONS:\n" + "\n".join(f"- {q}" for q in sub_queries) + "\n"
    user += f"\nANSWER:\n{answer}"

    try:
        data = chat_json(SYSTEM_PROMPT, user, provider=provider, max_tokens=400)
        report = CoverageReport(parts=data.get("parts"))
    except RateLimitExhausted:
        raise
    except Exception as e:
        print(f"[completeness] Coverage check failed ({e}); keeping the draft answer")
        return CoverageReport()

    print(f"[completeness] {len(report.parts)} part(s), missing: {report.missing or 'none'}")
    return report
