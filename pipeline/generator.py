"""
pipeline/generator.py
Stage 3 — LLM answer generation with inline citations.
"""

from __future__ import annotations
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a precise medical assistant. Answer the question using ONLY the provided context.

RULES:
1. Only use facts explicitly stated in the context. Never add outside knowledge.
2. Present the answer naturally in clear prose or bullets.
3. CRITICAL: Do NOT expose graph nodes, relationship names, Cypher, internal IDs, or graph triples in the final answer. Translate all graph relationships into natural language.
4. Add concise document filename + page citations at the end of each relevant claim or paragraph (e.g., [sample.pdf, p.4]).
5. If the context does not contain enough information, say: "Insufficient information in the provided sources."
6. Be concise and clinical. Avoid speculation.
7. End with a "Sources used:" section listing every unique citation you referenced."""

USER_TEMPLATE = """Context:
{context}

Question: {query}

Answer (with inline citations):"""


def generate_answer(
    query: str,
    fused_context: str,
    provider: str = "auto",
    stream: bool = False,
) -> str:
    """
    Generate a grounded answer using the fused context.

    Args:
        query:         Original user query
        fused_context: Output of reranker.fuse_and_rerank()
        provider:      "auto" | "groq" | "openai"
        stream:        Whether to stream the response (False for API use)

    Returns:
        Answer string with inline citations
    """
    prompt = USER_TEMPLATE.format(context=fused_context, query=query)

    if provider == "groq" or (provider == "auto" and os.getenv("GROQ_API_KEY")):
        return _groq_generate(prompt)
    else:
        return _openai_generate(prompt)


def _groq_generate(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def _openai_generate(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return resp.choices[0].message.content
