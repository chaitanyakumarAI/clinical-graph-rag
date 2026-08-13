"""
pipeline/vector_retriever.py
Stage 3 — Dense vector retrieval from Qdrant.
"""

from __future__ import annotations
from typing import List, Dict
from ingestion.embedder import embed_query
from vector.qdrant_client import QdrantStore

_store: QdrantStore | None = None


def _get_store() -> QdrantStore:
    global _store
    if _store is None:
        _store = QdrantStore()
    return _store


def retrieve_vector_context(query: str, top_k: int = 5) -> List[Dict]:
    """
    Embed the query and retrieve top-k relevant chunks from Qdrant.

    Returns:
        List of {chunk_id, text, source, page, score}
    """
    store     = _get_store()
    query_emb = embed_query(query)
    results   = store.search(query_emb, top_k=top_k)
    print(f"[vector_retriever] Retrieved {len(results)} chunks (top_k={top_k})")
    return results


def format_for_context(vector_results: List[Dict]) -> str:
    """Convert vector results into a context block for the LLM prompt."""
    if not vector_results:
        return ""
    lines = ["[DOCUMENT CONTEXT]"]
    for i, r in enumerate(vector_results, 1):
        citation = f"{r['source']} p.{r['page']} (score={r['score']:.2f})"
        lines.append(f"  [{i}] [{citation}]\n      {r['text'][:400]}")
    return "\n".join(lines)
