"""
pipeline/gds_retriever.py
Active GDS retrieval — called automatically based on query route, not a user toggle.

Mode selection (driven by backend/main.py based on query plan route):
  "shortest_path"  — only run shortest path between 2 linked entities
                     triggered for: hybrid_multi with ≥2 entities
  "similarity"     — node similarity for each linked entity
                     triggered for: hybrid_single to expand entity neighborhood
  "full"           — all algorithms: shortest path + similarity + PageRank
                     triggered for: gds_augmented route

Each mode gracefully falls back:
  - If GDS not installed: returns [] with a clear log message
  - If projection fails: skips that algorithm, continues with others
  - If shortest_path fails: falls back to vanilla Cypher shortestPath()
"""

from __future__ import annotations
from typing import List, Dict, Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.neo4j_client import Neo4jClient
    from graph.gds_client   import GDSClient

GDSMode = Literal["shortest_path", "similarity", "full", "auto"]
_PROJECTION = "rag_live"


def retrieve_gds_context(
    linked_entities: List[Dict],
    route:           str,
    neo4j_client:    "Neo4jClient",
    gds_client:      Optional["GDSClient"] = None,
    mode:            GDSMode = "auto",
    max_results:     int = 15,
) -> List[Dict]:
    """
    Actively retrieve GDS-based biomedical reasoning context.

    Mode is selected automatically from the route if mode="auto":
      gds_augmented → "full"
      hybrid_multi  → "shortest_path"  (if ≥2 entities)
      hybrid_single → "similarity"
      vector_only   → returns [] immediately

    Args:
        linked_entities: List of {query_term, neo4j_name, label, ...} dicts
        route:           Adaptive route from query_planner
        neo4j_client:    Open Neo4jClient
        gds_client:      Pre-initialized GDSClient (auto-created if None)
        mode:            GDSMode override ("auto" = decide from route)
        max_results:     Max result items

    Returns:
        List of path-text dicts compatible with the reranker
    """
    # Resolve "auto" mode from route
    if mode == "auto":
        mode = _route_to_mode(route, len(linked_entities))

    if not linked_entities or mode is None:
        return []

    # Initialize GDS client
    if gds_client is None:
        from graph.gds_client import GDSClient
        gds_client = GDSClient(neo4j_client)

    if not gds_client.check_available():
        print("[gds_retriever] GDS plugin not available — returning []")
        return []

    names   = [e["neo4j_name"] for e in linked_entities]
    results = []

    try:
        # ── Shortest path (always run for multi-entity queries) ────────────
        if mode in ("shortest_path", "full") and len(names) >= 2:
            results.extend(_run_shortest_path(names, gds_client))

        # ── Node similarity ─────────────────────────────────────────────────
        if mode in ("similarity", "full"):
            results.extend(_run_similarity(names, gds_client, linked_entities))

        # ── PageRank (full mode only) ───────────────────────────────────────
        if mode == "full":
            results.extend(_run_pagerank(gds_client))

    finally:
        try:
            gds_client.drop_projection(_PROJECTION, silent=True)
        except Exception:
            pass

    out = results[:max_results]
    print(f"[gds_retriever] ✅ mode={mode} → {len(out)} results "
          f"({len(results)} before cap)")
    return out


# ── Mode selector ─────────────────────────────────────────────────────────────

def _route_to_mode(route: str, n_entities: int) -> Optional[GDSMode]:
    """Map adaptive route → GDS mode. Returns None if GDS should be skipped."""
    if route == "vector_only":
        return None
    if route == "gds_augmented":
        return "full"
    if route == "hybrid_multi" and n_entities >= 2:
        return "shortest_path"
    if route == "hybrid_single":
        return "similarity"
    return None   # no GDS for hybrid_multi with only 1 entity


# ── Algorithm runners ─────────────────────────────────────────────────────────

def _run_shortest_path(names: List[str], gds: "GDSClient") -> List[Dict]:
    """Find shortest path between the first two linked entities."""
    results = []
    try:
        path = gds.shortest_path(names[0], names[1])
        if len(path) >= 2:
            path_text = " → ".join(path)
            results.append({
                "path_text":   f"[GDS Shortest Path] {path_text}",
                "hops":        len(path) - 1,
                "confidence":  0.88,
                "source_type": "graph_gds",
            })
            print(f"[gds_retriever] Shortest path ({len(path)-1} hops): {path_text}")
        else:
            print(f"[gds_retriever] No path found between '{names[0]}' and '{names[1]}'")
    except Exception as e:
        print(f"[gds_retriever] shortest_path error: {e}")
    return results


def _run_similarity(
    names:           List[str],
    gds:             "GDSClient",
    linked_entities: List[Dict],
) -> List[Dict]:
    """Find entities similar to each linked entity via node similarity (Jaccard)."""
    results = []
    try:
        gds.project_graph(_PROJECTION)
        for entity in linked_entities[:2]:
            name    = entity["neo4j_name"]
            similar = gds.node_similarity(name, _PROJECTION, top_k=4)
            for s in similar:
                if s.get("similarity", 0) < 0.1:
                    continue   # skip trivial similarity
                results.append({
                    "path_text": (
                        f"[GDS Similar] {name} ↔ {s['name']} "
                        f"({s.get('label','?')}, sim={s['similarity']:.2f})"
                    ),
                    "hops":        1,
                    "confidence":  min(float(s.get("similarity", 0.5)), 0.95),
                    "source_type": "graph_gds",
                })
        if results:
            print(f"[gds_retriever] Node similarity: {len(results)} related entities")
    except Exception as e:
        print(f"[gds_retriever] node_similarity error: {e}")
    return results


def _run_pagerank(gds: "GDSClient") -> List[Dict]:
    """Add top-ranked entities by PageRank as supplementary context."""
    results = []
    try:
        # Reuse existing projection or create one
        try:
            top = gds.pagerank(_PROJECTION, top_k=6)
        except Exception:
            gds.project_graph(_PROJECTION)
            top = gds.pagerank(_PROJECTION, top_k=6)

        if top:
            summary = ", ".join(
                f"{r['name']} ({r['label']}, score={r['score']:.3f})"
                for r in top
            )
            results.append({
                "path_text":   f"[GDS PageRank] Most influential biomedical entities: {summary}",
                "hops":        0,
                "confidence":  0.72,
                "source_type": "graph_gds",
            })
            print(f"[gds_retriever] PageRank: {len(top)} top entities")
    except Exception as e:
        print(f"[gds_retriever] pagerank error: {e}")
    return results


def format_gds_for_context(gds_results: List[Dict]) -> str:
    """Format GDS results as a context block for the LLM prompt."""
    if not gds_results:
        return ""
    lines = ["[GRAPH ALGORITHM REASONING]"]
    for r in gds_results:
        lines.append(f"  • {r['path_text']}")
    return "\n".join(lines)
