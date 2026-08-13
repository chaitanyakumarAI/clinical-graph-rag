"""
enrichment/primekg_loader.py
Stage 2 — Write PrimeKG matches and filtered edges into Neo4j.
"""

from __future__ import annotations
from typing import List, Dict
from graph.neo4j_client import Neo4jClient


def write_primekg_ids(match_results: List[Dict], client: Neo4jClient):
    """
    Tag matched Neo4j nodes with their PrimeKG canonical IDs.

    Args:
        match_results: Output of PrimeKGMatcher.match_all()
        client:        Open Neo4jClient
    """
    matched = [m for m in match_results if m.get("matched")]
    if not matched:
        print("[primekg_loader] No matched entities to tag.")
        return

    cypher = """
    UNWIND $rows AS row
    MATCH (n {name: row.neo4j_name})
    SET n.primekg_id    = row.primekg_id,
        n.primekg_name  = row.primekg_name,
        n.match_score   = row.score,
        n.match_method  = row.method
    """
    with client.driver.session() as s:
        s.run(cypher, rows=matched)

    print(f"[primekg_loader] ✅ Tagged {len(matched)} nodes with primekg_id")


def load_enriched_edges(edges: List[Dict], client: Neo4jClient):
    """
    Load Stage 2 filtered PrimeKG edges into Neo4j.

    Args:
        edges:  Output of primekg_edge_filter.filter_edges()
        client: Open Neo4jClient
    """
    if not edges:
        print("[primekg_loader] No edges to load.")
        return

    client.upsert_relations_batch(edges)
    print(f"[primekg_loader] ✅ {len(edges)} enriched edges loaded (stage='primekg_enriched')")
