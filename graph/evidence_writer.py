"""
graph/evidence_writer.py
Write explicit Evidence nodes that link relation edges to their supporting chunks.

Graph structure created:
    (:Entity)-[:TREATS {confidence}]->(:Entity)              ← existing edges
    (:Entity)-[:HAS_EVIDENCE {predicate}]->(:Evidence)       ← new
    (:Evidence {evidence_id, excerpt, confidence, ...})       ← new
    (:Evidence)-[:FROM_CHUNK]->(:Chunk {chunk_id, page})     ← new
    (:Evidence)-[:FROM_DOCUMENT]->(:Document {filename})     ← new

This makes provenance a first-class graph citizen:
  - "Which chunks support Metformin TREATS Diabetes?" →
      MATCH (:Drug {name:"Metformin"})-[:HAS_EVIDENCE {predicate:"TREATS"}]->(ev)
            -[:FROM_CHUNK]->(c) RETURN c.source, c.page, ev.excerpt

Evidence nodes are deduplicated by SHA-256(subject+predicate+object+chunk_id)[:16],
so re-ingesting a document is idempotent.

Called from graph/graph_loader.py after upsert_relations_batch().
"""

from __future__ import annotations
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.neo4j_client import Neo4jClient


def _evidence_id(subject: str, predicate: str, obj: str, chunk_id: str) -> str:
    key = f"{subject.lower()}|{predicate}|{obj.lower()}|{chunk_id}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Public API ────────────────────────────────────────────────────────────────

def write_evidence_batch(
    chunks:             List[Dict],
    relations:          List[Dict],
    extraction_results: List,           # List[ExtractionResult]
    client:             "Neo4jClient",
    extracted_at:       Optional[str] = None,
    batch_size:         int = 100,
) -> int:
    """
    Create Evidence nodes and link them to entity edges, chunks, and documents.

    Args:
        chunks:             Original chunk dicts (with chunk_id, text, source, page)
        relations:          Deduplicated relations (each has chunk_ids, subject, predicate, object)
        extraction_results: Raw extraction results (used for per-chunk confidence + excerpts)
        client:             Open Neo4jClient
        extracted_at:       ISO timestamp
        batch_size:         Cypher UNWIND batch size

    Returns:
        Number of Evidence nodes written
    """
    ts = extracted_at or _now_iso()

    # Build quick lookups
    chunk_by_id: Dict[str, Dict] = {c["chunk_id"]: c for c in chunks}
    # {(subj_lower, pred, obj_lower, chunk_id) → confidence}
    triple_chunk_conf = _build_triple_chunk_confidence(extraction_results)

    evidence_rows = []
    seen_ids: set = set()

    for rel in relations:
        subj = rel.get("subject", "")
        pred = rel.get("predicate", "")
        obj  = rel.get("object",   "")
        if not (subj and pred and obj):
            continue

        chunk_ids = rel.get("chunk_ids", [])
        if not chunk_ids:
            # No chunk provenance — still create one Evidence node without a chunk link
            eid = _evidence_id(subj, pred, obj, "")
            if eid not in seen_ids:
                seen_ids.add(eid)
                evidence_rows.append({
                    "evidence_id":  eid,
                    "subject":      subj,
                    "predicate":    pred,
                    "object":       obj,
                    "chunk_id":     None,
                    "excerpt":      "",
                    "confidence":   rel.get("confidence", 0.7),
                    "source":       (rel.get("sources") or ["unknown"])[0],
                    "page":         (rel.get("page_numbers") or [0])[0],
                    "extracted_at": ts,
                })
            continue

        for cid in chunk_ids:
            eid = _evidence_id(subj, pred, obj, cid)
            if eid in seen_ids:
                continue
            seen_ids.add(eid)

            chunk     = chunk_by_id.get(cid, {})
            conf_key  = (subj.lower(), pred, obj.lower(), cid)
            conf      = triple_chunk_conf.get(conf_key, rel.get("confidence", 0.7))
            text      = chunk.get("text", "")
            excerpt   = _best_excerpt(subj, obj, text, max_chars=200)

            evidence_rows.append({
                "evidence_id":  eid,
                "subject":      subj,
                "predicate":    pred,
                "object":       obj,
                "chunk_id":     cid,
                "excerpt":      excerpt,
                "confidence":   conf,
                "source":       chunk.get("source", ""),
                "page":         chunk.get("page", 0),
                "extracted_at": ts,
            })

    if not evidence_rows:
        print("[evidence_writer] No evidence rows to write")
        return 0

    _write_batches(evidence_rows, client, batch_size)
    print(f"[evidence_writer] ✅ {len(evidence_rows)} Evidence nodes written")
    return len(evidence_rows)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_triple_chunk_confidence(extraction_results: List) -> Dict:
    """
    Map (subj_lower, pred, obj_lower, chunk_id) → confidence
    from raw ExtractionResult objects.
    """
    mapping: Dict = {}
    for result in extraction_results:
        cid = result.chunk_id
        for rel in result.relations:
            key = (rel.subject.lower(), rel.predicate, rel.object.lower(), cid)
            mapping[key] = max(mapping.get(key, 0.0), rel.confidence)
    return mapping


def _best_excerpt(subject: str, obj: str, text: str, max_chars: int = 200) -> str:
    """
    Extract the most relevant sentence from chunk text as the evidence excerpt.
    Finds the sentence containing both subject and object names if possible,
    otherwise falls back to the first sentence.
    """
    if not text:
        return ""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    subj_l = subject.lower()
    obj_l  = obj.lower()
    for sent in sentences:
        low = sent.lower()
        if subj_l in low or obj_l in low:
            return (sent[:max_chars] + "…") if len(sent) > max_chars else sent
    return (sentences[0][:max_chars] + "…") if sentences and len(sentences[0]) > max_chars else (sentences[0] if sentences else "")


def _write_batches(rows: List[Dict], client: "Neo4jClient", batch_size: int):
    """Write Evidence nodes and all edges in batches via UNWIND."""
    with client.driver.session() as s:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            # 1. Create Evidence nodes
            s.run("""
                UNWIND $rows AS r
                MERGE (ev:Evidence {evidence_id: r.evidence_id})
                SET ev.subject      = r.subject,
                    ev.predicate    = r.predicate,
                    ev.object       = r.object,
                    ev.excerpt      = r.excerpt,
                    ev.confidence   = r.confidence,
                    ev.extracted_at = CASE WHEN ev.extracted_at IS NULL
                                       THEN r.extracted_at ELSE ev.extracted_at END,
                    ev.updated      = timestamp()
            """, rows=batch)

            # 2. Link subject entity → Evidence (HAS_EVIDENCE)
            s.run("""
                UNWIND $rows AS r
                MATCH (subj {name: r.subject})
                MATCH (ev:Evidence {evidence_id: r.evidence_id})
                MERGE (subj)-[he:HAS_EVIDENCE]->(ev)
                SET he.predicate = r.predicate
            """, rows=batch)

            # 3. Link Evidence → Chunk (FROM_CHUNK) — only if chunk_id is set
            chunk_rows = [r for r in batch if r.get("chunk_id")]
            if chunk_rows:
                s.run("""
                    UNWIND $rows AS r
                    MATCH (ev:Evidence {evidence_id: r.evidence_id})
                    MATCH (c:Chunk {chunk_id: r.chunk_id})
                    MERGE (ev)-[:FROM_CHUNK]->(c)
                """, rows=chunk_rows)

            # 4. Link Evidence → Document (FROM_DOCUMENT)
            doc_rows = [r for r in batch if r.get("source")]
            if doc_rows:
                s.run("""
                    UNWIND $rows AS r
                    MATCH (ev:Evidence {evidence_id: r.evidence_id})
                    MERGE (d:Document {filename: r.source})
                    MERGE (ev)-[:FROM_DOCUMENT]->(d)
                """, rows=doc_rows)
