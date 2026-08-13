"""
pipeline/reranker.py
Stage 3 / Phase 4 — Fuse and rerank graph paths + vector chunks + GDS results.

Phase 4 adds:
  - gds_results as a third source type ("graph_gds")
  - GDS paths are scored and reranked alongside document and graph sources
"""

from __future__ import annotations
import os
from typing import List, Dict, Tuple

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        print(f"[reranker] Loading model: {RERANKER_MODEL} …")
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
    return _reranker


def _score_candidates(query: str, candidates: List[str]) -> List[float]:
    """Score each candidate text against the query using the cross-encoder."""
    try:
        reranker = _get_reranker()
        pairs    = [[query, c] for c in candidates]
        scores   = reranker.compute_score(pairs, normalize=True)
        return scores if isinstance(scores, list) else [scores]
    except Exception as e:
        print(f"[reranker] ⚠️  Reranker unavailable ({e}), using original scores")
        return [0.5] * len(candidates)


def fuse_and_rerank(
    query:          str,
    graph_results:  List[Dict],
    vector_results: List[Dict],
    gds_results:    List[Dict] = [],   # Phase 4: GDS algorithm context
    top_n:          int = 8,
) -> Tuple[List[Dict], str]:
    """
    Merge graph paths, vector chunks, and GDS results; rerank; deduplicate.

    Args:
        query:          Original user query
        graph_results:  Output of graph_retriever.retrieve_graph_context()
        vector_results: Output of vector_retriever.retrieve_vector_context()
        gds_results:    Phase 4 — output of gds_retriever.retrieve_gds_context()
        top_n:          Number of top items to return

    Returns:
        (ranked_items, fused_context_string)
    """
    candidates = []

    # Normalize vector results (type = "document")
    for r in vector_results:
        candidates.append({
            "type":   "document",
            "text":   r["text"],
            "meta":   f"[Doc: {r['source']} p.{r['page']}]",
            "score":  r.get("score", 0.5),
            "source": r.get("source", ""),
            "page":   r.get("page", 0),
        })

    # Normalize graph results (type = "graph")
    for r in graph_results:
        candidates.append({
            "type":       "graph",
            "text":       r["path_text"],
            "meta":       f"[Graph] {r['path_text']}",
            "score":      r.get("confidence", 0.8),
            "confidence": r.get("confidence", 0.8),
            "pages":      r.get("pages", []),
            # Provenance fields from intent-aware retrieval
            "doc_source": r.get("doc_source", ""),
            "doc_page":   r.get("doc_page",   0),
            "excerpt":    r.get("excerpt",    ""),
        })

    # Phase 4: normalize GDS results (type = "graph_gds")
    for r in gds_results:
        candidates.append({
            "type":   "graph_gds",
            "text":   r["path_text"],
            "meta":   f"[GDS] {r['path_text']}",
            "score":  r.get("confidence", 0.75),
        })

    if not candidates:
        return [], ""

    # Deduplicate by text prefix (first 100 chars)
    seen_prefixes = set()
    unique = []
    for c in candidates:
        prefix = c["text"][:100].lower().strip()
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            unique.append(c)

    # Rerank all unique candidates
    texts         = [c["text"] for c in unique]
    rerank_scores = _score_candidates(query, texts)

    # Combine original score with rerank score (weighted)
    for item, rscore in zip(unique, rerank_scores):
        item["final_score"] = 0.4 * item["score"] + 0.6 * rscore

    # Sort descending and take top-N
    ranked = sorted(unique, key=lambda x: x["final_score"], reverse=True)[:top_n]

    # Build fused context string with source labels
    context_parts = []
    for item in ranked:
        context_parts.append(f"{item['meta']}\n{item['text']}")

    fused_context = "\n\n".join(context_parts)

    source_counts = {
        "document": sum(1 for r in ranked if r["type"] == "document"),
        "graph":    sum(1 for r in ranked if r["type"] == "graph"),
        "gds":      sum(1 for r in ranked if r["type"] == "graph_gds"),
    }
    print(f"[reranker] {len(candidates)} candidates → {len(ranked)} ranked "
          f"(doc={source_counts['document']}, graph={source_counts['graph']}, "
          f"gds={source_counts['gds']})")
    return ranked, fused_context
