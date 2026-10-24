"""
graph/graph_loader.py
Unified graph loader — builds the full Document → Chunk → Entity provenance graph.

After Phase 1-6 additions, the Neo4j graph now mirrors Qdrant's document structure:

    (:Document)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS {confidence}]->(:Entity)
                                      (:Entity)-[RELATION]->(:Entity)

This means every answer can be traced back to:
  entity → which chunks mentioned it → which page → which document

load_to_graph() is the single entry-point that replaces the old load_to_neo4j().
load_to_neo4j() is kept as a compatibility alias.
"""

from __future__ import annotations
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.neo4j_client import Neo4jClient
    from ingestion.entity_extractor import ExtractionResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Public entry-point ────────────────────────────────────────────────────────

def load_to_graph(
    chunks:              List[Dict],
    entities:            List[Dict],
    relations:           List[Dict],
    extraction_results:  List["ExtractionResult"],
    client:              "Neo4jClient",
    filename:            str = "",
    sha256:              str = "",       # canonical document identity
    stage:               str = "base_graph",
    extracted_at:        Optional[str] = None,
) -> Dict[str, int]:
    """
    Write the complete Document → Chunk → Entity provenance graph to Neo4j.

    Steps (in order):
      1. Create/update indexes
      2. Upsert Entity nodes (with provenance)
      3. Upsert Relation edges
      4. Upsert Chunk nodes
      5. Link Document → Chunks  (HAS_CHUNK)
      6. Link Chunks → Entities  (MENTIONS, per-chunk confidence)

    Args:
        chunks:             Original chunks from chunker (with text, page, etc.)
        entities:           Deduplicated entities from deduplicator.get_entities()
        relations:          Deduplicated relations from deduplicator.get_relations()
        extraction_results: Raw ExtractionResult list (for per-chunk entity confidence)
        client:             Open Neo4jClient
        filename:           Source PDF filename (used for Document node)
        stage:              Graph stage label
        extracted_at:       ISO timestamp (defaults to now)

    Returns:
        {"entities": N, "relations": M, "chunks": K, "mentions": P}
    """
    ts = extracted_at or _now_iso()

    # Stamp stage
    for e in entities:
        e.setdefault("stage", stage)
    for r in relations:
        r.setdefault("stage", stage)

    # 1. Ensure indexes
    client.create_indexes()

    # 2. Entity nodes
    client.upsert_entities_batch(entities, extracted_at=ts)
    print(f"[graph_loader] ✅ {len(entities)} entity nodes")

    # 3. Relation edges
    client.upsert_relations_batch(relations, extracted_at=ts)
    print(f"[graph_loader] ✅ {len(relations)} relation edges")

    # 4. Chunk nodes (shared structure between Neo4j and Qdrant)
    client.upsert_chunks_batch(chunks, extracted_at=ts)
    print(f"[graph_loader] ✅ {len(chunks)} chunk nodes")

    # 5. Document → Chunk edges (keyed by sha256 — the canonical identity)
    chunk_ids = [c["chunk_id"] for c in chunks]
    doc_name  = filename or (chunks[0]["source"] if chunks else "unknown")
    if sha256:
        client.link_chunks_to_document(sha256, chunk_ids, filename=doc_name)
    else:
        # Fallback for legacy callers that don't pass sha256
        client.link_chunks_to_document(doc_name, chunk_ids, filename=doc_name)
    print(f"[graph_loader] ✅ Document '{doc_name}' → {len(chunk_ids)} chunks")

    # 6. Chunk → Entity MENTIONS edges
    chunk_entity_conf = _build_chunk_entity_confidence(extraction_results)
    mention_count = client.link_chunks_to_entities(chunk_entity_conf)
    print(f"[graph_loader] ✅ {mention_count} MENTIONS edges (Chunk→Entity)")

    # 7. Evidence nodes (explicit edge → chunk provenance graph)
    evidence_count = 0
    try:
        from graph.evidence_writer import write_evidence_batch
        evidence_count = write_evidence_batch(
            chunks             = chunks,
            relations          = relations,
            extraction_results = extraction_results,
            client             = client,
            extracted_at       = ts,
            doc_sha256         = sha256,
        )
    except Exception as e:
        print(f"[graph_loader] ⚠️ Evidence writing failed (non-fatal): {e}")

    return {
        "entities":  len(entities),
        "relations": len(relations),
        "chunks":    len(chunks),
        "mentions":  mention_count,
        "evidence":  evidence_count,
    }


def load_chunks_to_qdrant(
    chunks:     List[Dict],
    collection: str = "medical_chunks",
) -> int:
    """Store embedded chunks in Qdrant (unchanged from before)."""
    from vector.qdrant_client import QdrantStore
    store = QdrantStore(collection=collection)
    store.upsert_chunks(chunks)
    print(f"[graph_loader] ✅ Qdrant: {len(chunks)} chunks → '{collection}'")
    return len(chunks)


# ── Compatibility alias ───────────────────────────────────────────────────────

def load_to_neo4j(
    entities:    List[Dict],
    relations:   List[Dict],
    client:      "Neo4jClient",
    stage:       str = "base_graph",
    extracted_at: Optional[str] = None,
) -> Dict[str, int]:
    """
    Legacy compatibility wrapper — loads entities and relations without Chunk nodes.
    Prefer load_to_graph() for new code.
    """
    ts = extracted_at or _now_iso()
    for e in entities:
        e.setdefault("stage", stage)
    for r in relations:
        r.setdefault("stage", stage)

    client.create_indexes()
    client.upsert_entities_batch(entities, extracted_at=ts)
    client.upsert_relations_batch(relations, extracted_at=ts)

    print(f"[graph_loader] ✅ Neo4j (legacy): {len(entities)} entities, {len(relations)} relations")
    return {"entities": len(entities), "relations": len(relations)}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_chunk_entity_confidence(
    extraction_results: List["ExtractionResult"],
) -> Dict[str, Dict[str, float]]:
    """
    Build a map: {chunk_id → {entity_name → confidence}}
    from raw extraction results, for use when creating MENTIONS edges.

    When the same entity appears in a chunk multiple times across results,
    the maximum confidence is kept.
    """
    # chunk_id → entity_name → max_confidence
    mapping: Dict[str, Dict[str, float]] = defaultdict(dict)

    for result in extraction_results:
        cid = result.chunk_id
        for entity in result.entities:
            name = entity.name.strip()
            if not name:
                continue
            existing = mapping[cid].get(name, 0.0)
            mapping[cid][name] = max(existing, entity.confidence)

    return dict(mapping)
