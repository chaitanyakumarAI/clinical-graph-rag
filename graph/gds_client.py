"""
graph/gds_client.py
Phase 4 — Neo4j Graph Data Science (GDS) algorithm wrapper.

Requires GDS plugin: verify with `CALL gds.list()` in Neo4j browser.
If GDS is not installed, all methods raise GDSUnavailableError.

Supported algorithms:
  - pagerank()         → influence ranking of biomedical entities
  - shortest_path()    → connecting path between two entities
  - community()        → Louvain community detection (cluster entities)
  - node_similarity()  → entities similar to a given node
  - centrality()       → betweenness centrality (hub entities)
"""

from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.neo4j_client import Neo4jClient


class GDSUnavailableError(RuntimeError):
    """Raised when the Neo4j GDS plugin is not installed."""


def _check_gds(client: "Neo4jClient"):
    """Raise GDSUnavailableError if GDS is not installed."""
    try:
        client.run_query("CALL gds.list() YIELD procName RETURN procName LIMIT 1")
    except Exception as e:
        raise GDSUnavailableError(
            "Neo4j GDS plugin not found. Install it via Neo4j Desktop → Plugins, "
            "or from https://neo4j.com/docs/graph-data-science/current/installation/"
        ) from e


class GDSClient:
    """
    Wrapper for Neo4j GDS algorithms.

    Usage:
        with Neo4jClient() as neo4j:
            gds = GDSClient(neo4j)
            gds.check_available()
            top = gds.pagerank("all_entities", top_k=10)
    """

    def __init__(self, client: "Neo4jClient"):
        self._client = client
        self._available: Optional[bool] = None

    def check_available(self) -> bool:
        """Return True if GDS plugin is installed, False otherwise."""
        if self._available is None:
            try:
                _check_gds(self._client)
                self._available = True
            except GDSUnavailableError:
                self._available = False
        return self._available

    # ── Graph Projection ──────────────────────────────────────────────────────

    def project_graph(
        self,
        projection_name: str,
        node_labels:     Optional[List[str]] = None,
        rel_types:       Optional[List[str]] = None,
    ) -> str:
        """
        Create an in-memory GDS graph projection.
        Returns the projection name on success.
        """
        labels = node_labels or ["*"]   # all labels
        rels   = rel_types   or ["*"]   # all relationship types

        # Drop if already exists
        self.drop_projection(projection_name, silent=True)

        query = """
        CALL gds.graph.project($name, $labels, $rels)
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        rows = self._client.run_query(query, {
            "name":   projection_name,
            "labels": labels,
            "rels":   rels,
        })
        if rows:
            r = rows[0]
            print(f"[gds] Projected '{projection_name}': "
                  f"{r['nodeCount']} nodes, {r['relationshipCount']} relationships")
        return projection_name

    def drop_projection(self, projection_name: str, silent: bool = False):
        """Drop a graph projection."""
        try:
            self._client.run_query(
                "CALL gds.graph.drop($name) YIELD graphName",
                {"name": projection_name},
            )
        except Exception:
            if not silent:
                raise

    # ── PageRank ──────────────────────────────────────────────────────────────

    def pagerank(
        self,
        projection_name: str,
        top_k: int = 20,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> List[Dict]:
        """
        Run PageRank to find most influential entities in the graph.

        Returns:
            List of {name, label, score} dicts sorted by score descending
        """
        query = """
        CALL gds.pageRank.stream($projection, {
            dampingFactor: $damping,
            maxIterations: $iterations
        })
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.name AS name,
               labels(node)[0] AS label,
               round(score, 4) AS score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self._client.run_query(query, {
            "projection": projection_name,
            "damping":    damping,
            "iterations": iterations,
            "top_k":      top_k,
        })

    # ── Shortest Path ─────────────────────────────────────────────────────────

    def shortest_path(
        self,
        start_name:  str,
        end_name:    str,
        max_depth:   int = 5,
    ) -> List[str]:
        """
        Find the shortest path between two named entities.

        Returns:
            Ordered list of entity names along the path, or [] if no path found
        """
        query = """
        MATCH (start {name: $start}), (end {name: $end})
        CALL gds.shortestPath.dijkstra.stream('all_entities', {
            sourceNode: id(start),
            targetNode: id(end)
        })
        YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
        UNWIND gds.util.asNodes(nodeIds) AS node
        RETURN node.name AS name
        """
        try:
            rows = self._client.run_query(query, {
                "start": start_name,
                "end":   end_name,
            })
            return [r["name"] for r in rows]
        except Exception:
            # Fallback: use vanilla Cypher shortest path (no GDS needed)
            fallback = """
            MATCH path = shortestPath((a {name: $start})-[*1..6]-(b {name: $end}))
            RETURN [n IN nodes(path) | n.name] AS names
            LIMIT 1
            """
            rows = self._client.run_query(fallback, {
                "start": start_name, "end": end_name,
            })
            return rows[0]["names"] if rows else []

    # ── Community Detection ───────────────────────────────────────────────────

    def community(
        self,
        projection_name: str,
        top_communities: int = 10,
    ) -> Dict[str, int]:
        """
        Run Louvain community detection.

        Returns:
            Dict mapping entity_name → community_id
        """
        query = """
        CALL gds.louvain.stream($projection)
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId) AS node, communityId
        RETURN node.name AS name, communityId
        ORDER BY communityId ASC
        LIMIT 500
        """
        rows = self._client.run_query(query, {"projection": projection_name})
        return {r["name"]: r["communityId"] for r in rows}

    # ── Node Similarity ───────────────────────────────────────────────────────

    def node_similarity(
        self,
        entity_name:    str,
        projection_name: str,
        top_k:          int = 10,
    ) -> List[Dict]:
        """
        Find entities most similar to a given entity using Jaccard similarity
        over shared neighbor sets.

        Returns:
            List of {name, label, similarity} dicts
        """
        query = """
        CALL gds.nodeSimilarity.stream($projection)
        YIELD node1, node2, similarity
        WITH gds.util.asNode(node1) AS n1,
             gds.util.asNode(node2) AS n2,
             similarity
        WHERE n1.name = $entity_name OR n2.name = $entity_name
        WITH CASE WHEN n1.name = $entity_name THEN n2 ELSE n1 END AS similar,
             similarity
        RETURN similar.name AS name,
               labels(similar)[0] AS label,
               round(similarity, 4) AS similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        return self._client.run_query(query, {
            "projection":  projection_name,
            "entity_name": entity_name,
            "top_k":       top_k,
        })

    # ── Betweenness Centrality ────────────────────────────────────────────────

    def centrality(
        self,
        projection_name: str,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Find hub entities by betweenness centrality (nodes on the most paths).

        Returns:
            List of {name, label, score} dicts sorted by score descending
        """
        query = """
        CALL gds.betweenness.stream($projection)
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS node, score
        RETURN node.name AS name,
               labels(node)[0] AS label,
               round(score, 4) AS score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self._client.run_query(query, {
            "projection": projection_name,
            "top_k":      top_k,
        })
