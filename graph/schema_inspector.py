"""
graph/schema_inspector.py
Phase 3 — Live Neo4j schema introspection with TTL cache.

Queries the running graph for all current node labels and relationship types.
Used by cypher_generator.py to build accurate LLM prompts.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.neo4j_client import Neo4jClient

_CACHE_TTL_SECONDS = 300  # 5-minute cache

_cached_schema: Optional[Dict] = None
_cache_ts: float = 0.0


def get_live_schema(client: "Neo4jClient", force_refresh: bool = False) -> Dict:
    """
    Return a dict describing the current graph schema:
    {
      "node_labels":      ["Disease", "Drug", ...],
      "relation_types":   ["TREATS", "CAUSES", ...],
      "node_properties":  {"Disease": ["name","confidence","pages",...], ...},
    }

    Results are cached for TTL seconds to avoid hammering Neo4j.
    """
    global _cached_schema, _cache_ts

    if not force_refresh and _cached_schema and (time.time() - _cache_ts) < _CACHE_TTL_SECONDS:
        return _cached_schema

    schema: Dict = {
        "node_labels":     [],
        "relation_types":  [],
        "node_properties": {},
    }

    # Node labels
    try:
        rows = client.run_query("CALL db.labels() YIELD label RETURN label")
        schema["node_labels"] = [r["label"] for r in rows]
    except Exception:
        # Fallback: use the static list from neo4j_client
        from graph.neo4j_client import NODE_LABELS
        schema["node_labels"] = list(NODE_LABELS)

    # Relationship types
    try:
        rows = client.run_query(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        )
        schema["relation_types"] = [r["relationshipType"] for r in rows]
    except Exception:
        from graph.neo4j_client import RELATION_TYPES
        schema["relation_types"] = list(RELATION_TYPES)

    # Node properties (sample one node per label to infer properties)
    for label in schema["node_labels"]:
        try:
            rows = client.run_query(
                f"MATCH (n:{label}) RETURN keys(n) AS props LIMIT 1"
            )
            if rows:
                schema["node_properties"][label] = rows[0].get("props", [])
        except Exception:
            schema["node_properties"][label] = ["name", "confidence", "sources", "pages"]

    _cached_schema = schema
    _cache_ts = time.time()
    return schema


def format_schema_for_prompt(schema: Dict) -> str:
    """
    Render the schema as a compact text block for injection into LLM prompts.

    Example output:
        Node Labels: Disease, Drug, Gene, Protein, Pathway, ...
        Relationship Types: TREATS, CAUSES, TARGETS, REGULATES, ...
        Key Properties: name, confidence, sources, pages, extracted_at
    """
    labels = ", ".join(schema.get("node_labels", []))
    rels   = ", ".join(schema.get("relation_types", []))

    # Collect unique property names across all labels
    all_props: set = set()
    for props in schema.get("node_properties", {}).values():
        all_props.update(props)
    # Prioritize key provenance properties in display order
    priority = ["name", "confidence", "sources", "pages", "extracted_at",
                "umls_cui", "mesh_id", "rxnorm_id", "aliases", "stage"]
    sorted_props = [p for p in priority if p in all_props] + \
                   sorted(all_props - set(priority))

    return (
        f"Node Labels: {labels}\n"
        f"Relationship Types: {rels}\n"
        f"Key Node Properties: {', '.join(sorted_props[:12])}\n"
        f"Key Edge Properties: confidence, sources, page_numbers, extracted_at, stage"
    )
