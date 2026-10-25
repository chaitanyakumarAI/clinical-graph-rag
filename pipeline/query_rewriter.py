"""
pipeline/query_rewriter.py
Rewrite the user's question before retrieval.

Why:
  Users type questions like "side effects of metfromin?" or "BP meds safe for
  T2DM pts". Vector search and entity linking work much better on clean,
  explicit medical language, so we first rewrite the question:
    - fix spelling of drug / disease names
    - expand abbreviations (T2DM → Type 2 Diabetes, BP → blood pressure)
    - make the information need explicit

  Questions with several parts ("Which diabetes drugs interact with blood-pressure
  drugs, and why?") are also split into sub-queries. Each sub-query gets its own
  vector search, so one part of the question cannot crowd out the other in the
  top-k results — important when the answer is spread across several PDFs.

The original question is still what the final answer responds to; the rewrite
only steers retrieval.
"""

from __future__ import annotations
from typing import List

from pydantic import BaseModel, field_validator

from llm import RateLimitExhausted, chat_json

MAX_SUB_QUERIES = 3

SYSTEM_PROMPT = """You rewrite medical questions so a search engine over clinical documents finds the right passages.

Produce:
1. "rewritten": the same question, made explicit and searchable.
   - Fix spelling of drug and disease names; use generic drug names.
   - Expand abbreviations (T2DM -> Type 2 Diabetes, BP -> blood pressure, MI -> myocardial infarction, AF -> atrial fibrillation, CKD -> chronic kidney disease).
   - Keep the exact meaning. Do NOT answer the question, add facts, or add drugs/diseases the user did not mention.
2. "sub_queries": ONLY if the question has several distinct parts, or asks how two topics connect,
   split it into 2-3 short self-contained search queries. Otherwise return [].

Return ONLY JSON:
{"rewritten": "...", "sub_queries": ["...", "..."]}

Examples:
Q: side effects of metfromin?
{"rewritten": "What are the side effects of metformin?", "sub_queries": []}

Q: which T2DM drugs interact with BP meds and what happens
{"rewritten": "Which Type 2 Diabetes drugs interact with blood pressure medications, and what are the effects of these interactions?",
 "sub_queries": ["Type 2 Diabetes drugs and their drug interactions", "blood pressure medication drug interactions with diabetes drugs"]}"""


class RewrittenQuery(BaseModel):
    original:    str
    rewritten:   str
    sub_queries: List[str] = []

    @field_validator("sub_queries")
    @classmethod
    def limit_sub_queries(cls, v):
        return [q.strip() for q in v if isinstance(q, str) and q.strip()][:MAX_SUB_QUERIES]

    @property
    def search_queries(self) -> List[str]:
        """Every query to run vector search with: the rewrite first, then sub-queries."""
        queries: List[str] = []
        for q in [self.rewritten, *self.sub_queries]:
            if q.lower() not in (seen.lower() for seen in queries):
                queries.append(q)
        return queries


def no_rewrite(question: str) -> RewrittenQuery:
    """Pass-through used when rewriting is disabled."""
    return RewrittenQuery(original=question, rewritten=question)


def rewrite_query(question: str, provider: str = "auto") -> RewrittenQuery:
    """
    Rewrite a user question for retrieval.

    Falls back to the original question if the LLM reply is unusable, so a
    rewriting problem never breaks the query pipeline.
    """
    try:
        data = chat_json(SYSTEM_PROMPT, f"Q: {question}", provider=provider, max_tokens=300)
        rewritten = (data.get("rewritten") or "").strip() or question
        result = RewrittenQuery(
            original    = question,
            rewritten   = rewritten,
            sub_queries = data.get("sub_queries") or [],
        )
    except RateLimitExhausted:
        raise
    except Exception as e:
        print(f"[query_rewriter] Rewrite failed ({e}), using the original question")
        return no_rewrite(question)

    print(f"[query_rewriter] '{question}' → '{result.rewritten}'"
          + (f" + {len(result.sub_queries)} sub-queries" if result.sub_queries else ""))
    return result


if __name__ == "__main__":
    for q in [
        "side effects of metfromin?",
        "BP meds safe for T2DM pts with protein in urine",
        "Which diabetes drugs interact with blood-pressure medications?",
    ]:
        r = rewrite_query(q)
        print(f"{q}\n  → {r.rewritten}\n  sub: {r.sub_queries}\n")
