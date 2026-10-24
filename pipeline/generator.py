"""
pipeline/generator.py
Stage 3 — LLM answer generation with inline citations.
"""

from __future__ import annotations

from llm import chat

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
    query:         str,
    fused_context: str,
    provider:      str = "auto",
) -> str:
    """
    Generate a grounded answer using the fused context.

    Args:
        query:         Original user query
        fused_context: Output of reranker.fuse_and_rerank()
        provider:      "auto" | "gemini" | "groq" | "openai" | "ollama"

    Returns:
        Answer string with inline citations
    """
    prompt = USER_TEMPLATE.format(context=fused_context, query=query)
    return chat(SYSTEM_PROMPT, prompt, provider=provider,
                max_tokens=1024, temperature=0.1)
