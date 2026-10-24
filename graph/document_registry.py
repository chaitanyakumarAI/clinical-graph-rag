"""
graph/document_registry.py
Phase 6 — Track all ingested documents to prevent duplicate processing.

Each ingested PDF gets a Document node in Neo4j:
  {:Document {sha256, filename, ingested_at, page_count, entity_count, relation_count}}

The /ingest endpoint checks this before processing — if the same file has
already been ingested (same SHA-256), it returns immediately without re-processing.
"""

from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.neo4j_client import Neo4jClient


def compute_sha256(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def is_already_ingested(sha256: str, client: "Neo4jClient") -> bool:
    """
    Return True if a document with this SHA-256 hash has already been ingested.
    """
    rows = client.run_query(
        "MATCH (d:Document {sha256: $sha256}) RETURN d.filename AS filename LIMIT 1",
        {"sha256": sha256},
    )
    return bool(rows)


def register_document(
    sha256:         str,
    filename:       str,
    client:         "Neo4jClient",
    page_count:     int = 0,
    entity_count:   int = 0,
    relation_count: int = 0,
) -> Dict:
    """
    Create or update a Document node for a processed file.

    Returns the document metadata dict.
    """
    ingested_at = datetime.now(timezone.utc).isoformat()
    cypher = """
    MERGE (d:Document {sha256: $sha256})
    SET d.filename       = $filename,
        d.ingested_at    = $ingested_at,
        d.page_count     = $page_count,
        d.entity_count   = $entity_count,
        d.relation_count = $relation_count,
        d.updated        = timestamp()
    RETURN d.sha256 AS sha256, d.filename AS filename, d.ingested_at AS ingested_at
    """
    client.run_query(cypher, {
        "sha256":         sha256,
        "filename":       filename,
        "ingested_at":    ingested_at,
        "page_count":     page_count,
        "entity_count":   entity_count,
        "relation_count": relation_count,
    })
    print(f"[document_registry] ✅ Registered: {filename} (sha256={sha256[:12]}…)")
    return {
        "sha256":         sha256,
        "filename":       filename,
        "ingested_at":    ingested_at,
        "page_count":     page_count,
        "entity_count":   entity_count,
        "relation_count": relation_count,
    }


def list_documents(client: "Neo4jClient") -> list:
    """Return all properly-registered documents (sha256-keyed nodes only)."""
    return client.run_query(
        """
        MATCH (d:Document)
        WHERE d.sha256 IS NOT NULL
        RETURN d.filename    AS filename,
               d.sha256      AS sha256,
               d.ingested_at AS ingested_at,
               d.page_count  AS pages,
               d.entity_count  AS entities,
               d.relation_count AS relations,
               'ingested'    AS status
        ORDER BY d.ingested_at DESC
        """
    )


def delete_document(sha256: str, client: "Neo4jClient") -> Dict:
    """
    Remove a document's Neo4j data identified by its SHA-256 hash.

    Strategy:
      1. Look up the Document node to retrieve filename.
      2. Delete Chunk nodes that are linked ONLY to this document
         (i.e. not shared with any other Document via HAS_CHUNK).
      3. Detach-delete the Document node itself.

    Entity and Relationship nodes shared across documents are intentionally
    left in place — they carry provenance from multiple sources.

    Returns:
        dict with filename, chunks_deleted, doc_deleted (bool), error (str|None)
    """
    # Step 1: find the document
    rows = client.run_query(
        "MATCH (d:Document {sha256: $sha256}) RETURN d.filename AS filename LIMIT 1",
        {"sha256": sha256},
    )
    if not rows:
        return {"filename": None, "chunks_deleted": 0, "doc_deleted": False,
                "error": "Document not found"}

    filename = rows[0]["filename"]

    # Step 2: delete Chunk nodes exclusively owned by this document
    del_chunks_result = client.run_query(
        """
        MATCH (d:Document {sha256: $sha256})-[:HAS_CHUNK]->(c:Chunk)
        WHERE NOT EXISTS {
            MATCH (other:Document)-[:HAS_CHUNK]->(c)
            WHERE other.sha256 <> $sha256
        }
        WITH c, count(c) AS cnt
        DETACH DELETE c
        RETURN count(*) AS deleted
        """,
        {"sha256": sha256},
    )
    chunks_deleted = del_chunks_result[0]["deleted"] if del_chunks_result else 0

    # Step 3: detach-delete the Document node itself
    client.run_query(
        "MATCH (d:Document {sha256: $sha256}) DETACH DELETE d",
        {"sha256": sha256},
    )

    print(f"[document_registry] 🗑️  Deleted: {filename} "
          f"(sha256={sha256[:12]}…, chunks={chunks_deleted})")
    return {
        "filename":      filename,
        "chunks_deleted": chunks_deleted,
        "doc_deleted":   True,
        "error":         None,
    }


def purge_legacy_documents(client: "Neo4jClient") -> Dict:
    """
    Remove all Document nodes that have no sha256 (created by old code that
    keyed Document nodes by filename only). These are stale duplicates that
    pre-date the SHA-256 deduplication system.

    Safe to call at any time — sha256-keyed nodes are never touched.

    Returns:
        {"purged": N}  — number of legacy nodes removed
    """
    result = client.run_query(
        """
        MATCH (d:Document)
        WHERE d.sha256 IS NULL
        DETACH DELETE d
        RETURN count(*) AS purged
        """
    )
    purged = result[0]["purged"] if result else 0
    print(f"[document_registry] 🧹 Purged {purged} legacy Document node(s) (no sha256)")
    return {"purged": purged}


def get_source_map(client: "Neo4jClient") -> Dict[str, str]:
    """Map internal chunk source IDs (tmpXXX) to actual filenames."""
    rows = client.run_query(
        "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN DISTINCT c.source AS tmp_id, d.filename AS filename"
    )
    return {r["tmp_id"]: r["filename"] for r in rows if r.get("tmp_id") and r.get("filename")}
