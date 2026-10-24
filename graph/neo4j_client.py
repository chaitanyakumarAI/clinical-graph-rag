"""
graph/neo4j_client.py
Phase 1 Enhancement — Full provenance tracking + expanded schema.

Every node stores:
  confidence, extracted_at (ISO timestamp), chunk_ids, pages

Every relationship stores:
  confidence, extracted_at, page_numbers, paragraph_ids, chunk_ids

Phase 2 hooks are pre-wired (umls_cui, mesh_id, rxnorm_id, valid_from, valid_until)
but not populated until Phase 2 identifier_linker runs.
"""

from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Full expanded node label list (Phase 1)
NODE_LABELS = [
    # Clinical
    "Disease", "Drug", "Symptom", "Treatment",
    "SideEffect", "PatientGroup", "Organ", "MedicalGuideline",
    # Molecular (Phase 1 additions)
    "Gene", "Protein", "Pathway", "Compound",
    "CellType", "MolecularFunction", "BiologicalProcess",
]

# Full expanded relationship list (Phase 1)
RELATION_TYPES = [
    "TREATS", "CAUSES", "INTERACTS_WITH", "HAS_SYMPTOM",
    "CONTRAINDICATED_FOR", "AFFECTS_ORGAN", "RECOMMENDED_BY",
    "TARGETS", "REGULATES", "INHIBITS", "ACTIVATES",
    "LEADS_TO", "BIOMARKER_FOR", "EXPRESSED_IN", "PARTICIPATES_IN",
    "RELATED_TO",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Connection ────────────────────────────────────────────────────────────

    def verify_connection(self) -> str:
        with self.driver.session() as s:
            return s.run("RETURN 'Connection OK' AS status").single()["status"]

    # ── Schema ────────────────────────────────────────────────────────────────

    def create_indexes(self):
        """Create indexes on name property for all entity labels, plus provenance indexes."""
        with self.driver.session() as s:
            for label in NODE_LABELS:
                s.run(
                    f"CREATE INDEX {label.lower()}_name IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.name)"
                )
            # Phase 2 hooks: canonical ID indexes (created now, populated later)
            for label in ["Drug", "Disease", "Gene", "Protein"]:
                s.run(
                    f"CREATE INDEX {label.lower()}_umls IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.umls_cui)"
                )
                s.run(
                    f"CREATE INDEX {label.lower()}_mesh IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.mesh_id)"
                )
            for label in ["Drug", "Compound"]:
                s.run(
                    f"CREATE INDEX {label.lower()}_rxnorm IF NOT EXISTS "
                    f"FOR (n:{label}) ON (n.rxnorm_id)"
                )
            # Document registry index (Phase 6)
            s.run(
                "CREATE INDEX document_hash IF NOT EXISTS "
                "FOR (n:Document) ON (n.sha256)"
            )
            # Chunk node index (provenance graph)
            s.run(
                "CREATE INDEX chunk_id_idx IF NOT EXISTS "
                "FOR (n:Chunk) ON (n.chunk_id)"
            )
            s.run(
                "CREATE INDEX chunk_source_idx IF NOT EXISTS "
                "FOR (n:Chunk) ON (n.source)"
            )
            # Evidence node indexes (explicit edge provenance)
            s.run(
                "CREATE INDEX evidence_id_idx IF NOT EXISTS "
                "FOR (n:Evidence) ON (n.evidence_id)"
            )
            s.run(
                "CREATE INDEX evidence_pred_idx IF NOT EXISTS "
                "FOR (n:Evidence) ON (n.predicate)"
            )
        print(f"[neo4j] ✅ Indexes ensured for {len(NODE_LABELS)} labels + Chunk + Evidence")


    # ── Upsert helpers ────────────────────────────────────────────────────────

    def upsert_entity(self, entity: Dict[str, Any], extracted_at: Optional[str] = None):
        """
        MERGE a single entity node with full provenance.

        entity dict keys:
            type (str), name (str), aliases (list), sources (list),
            confidence (float), chunk_ids (list), pages (list), stage (str)
            umls_cui, mesh_id, rxnorm_id, drugbank_id  ← Phase 2 (optional)
        """
        label = entity["type"]
        ts    = extracted_at or _now_iso()
        cypher = f"""
        MERGE (n:{label} {{name: $name}})
        SET n.aliases      = $aliases,
            n.sources      = $sources,
            n.confidence   = CASE WHEN n.confidence IS NULL
                               THEN $confidence
                               ELSE (n.confidence + $confidence) / 2.0 END,
            n.chunk_ids    = apoc.coll.toSet(
                                coalesce(n.chunk_ids, []) + $chunk_ids),
            n.pages        = apoc.coll.toSet(
                                coalesce(n.pages, []) + $pages),
            n.stage        = $stage,
            n.extracted_at = CASE WHEN n.extracted_at IS NULL
                               THEN $extracted_at ELSE n.extracted_at END,
            n.updated      = timestamp()
        """
        with self.driver.session() as s:
            s.run(cypher, {
                "name":        entity["name"],
                "aliases":     entity.get("aliases",    []),
                "sources":     entity.get("sources",    []),
                "confidence":  entity.get("confidence", 0.8),
                "chunk_ids":   entity.get("chunk_ids",  []),
                "pages":       [entity.get("page", 0)] if entity.get("page") else
                               entity.get("pages", []),
                "stage":       entity.get("stage",      "base_graph"),
                "extracted_at": ts,
            })

    def upsert_entities_batch(
        self,
        entities: List[Dict[str, Any]],
        extracted_at: Optional[str] = None,
    ):
        """Batch upsert entities, grouped by label for efficiency."""
        from collections import defaultdict
        ts = extracted_at or _now_iso()
        by_label = defaultdict(list)
        for e in entities:
            by_label[e["type"]].append(e)

        with self.driver.session() as s:
            for label, batch in by_label.items():
                rows = [
                    {
                        "name":        e["name"],
                        "aliases":     e.get("aliases",    []),
                        "sources":     e.get("sources",    []),
                        "confidence":  e.get("confidence", 0.8),
                        "chunk_ids":   e.get("chunk_ids",  []),
                        "pages":       e.get("pages",      []),
                        "stage":       e.get("stage",      "base_graph"),
                        "extracted_at": ts,
                    }
                    for e in batch
                ]
                # Note: apoc.coll.toSet requires APOC plugin.
                # If APOC is not available, fall back to simple SET (no accumulation).
                cypher = f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{name: row.name}})
                SET n.aliases      = row.aliases,
                    n.sources      = row.sources,
                    n.confidence   = CASE WHEN n.confidence IS NULL
                                       THEN row.confidence
                                       ELSE (n.confidence + row.confidence) / 2.0 END,
                    n.chunk_ids    = row.chunk_ids,
                    n.pages        = row.pages,
                    n.stage        = row.stage,
                    n.extracted_at = CASE WHEN n.extracted_at IS NULL
                                       THEN row.extracted_at ELSE n.extracted_at END,
                    n.updated      = timestamp()
                """
                s.run(cypher, rows=rows)

        print(f"[neo4j] ✅ Upserted {len(entities)} entities "
              f"({len(by_label)} labels) at {ts}")

    def upsert_relation(
        self,
        relation: Dict[str, Any],
        extracted_at: Optional[str] = None,
    ):
        """
        MERGE a single directed relationship with full provenance.

        relation dict keys:
            subject_type, subject, predicate, object_type, object,
            confidence, chunk_ids (list), sources (list), stage,
            page_numbers (list), paragraph_ids (list)
            valid_from, valid_until  ← Phase 2 temporal (optional)
        """
        subj_label = relation["subject_type"]
        obj_label  = relation["object_type"]
        rel_type   = relation["predicate"]
        ts         = extracted_at or _now_iso()

        cypher = f"""
        MERGE (a:{subj_label} {{name: $subj}})
        MERGE (b:{obj_label}  {{name: $obj}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.confidence    = $confidence,
            r.chunk_ids     = $chunk_ids,
            r.sources       = $sources,
            r.page_numbers  = $page_numbers,
            r.paragraph_ids = $paragraph_ids,
            r.stage         = $stage,
            r.extracted_at  = CASE WHEN r.extracted_at IS NULL
                                THEN $extracted_at ELSE r.extracted_at END,
            r.updated       = timestamp()
        """
        with self.driver.session() as s:
            s.run(cypher, {
                "subj":          relation["subject"],
                "obj":           relation["object"],
                "confidence":    relation.get("confidence",    0.8),
                "chunk_ids":     relation.get("chunk_ids",     []),
                "sources":       relation.get("sources",       []),
                "page_numbers":  relation.get("page_numbers",  []),
                "paragraph_ids": relation.get("paragraph_ids", []),
                "stage":         relation.get("stage",         "base_graph"),
                "extracted_at":  ts,
            })

    def upsert_relations_batch(
        self,
        relations: List[Dict[str, Any]],
        batch_size: int = 200,
        extracted_at: Optional[str] = None,
    ):
        """Batch upsert relations with full provenance."""
        from collections import defaultdict
        ts = extracted_at or _now_iso()
        grouped = defaultdict(list)
        for r in relations:
            key = (r["subject_type"], r["predicate"], r["object_type"])
            grouped[key].append(r)

        with self.driver.session() as s:
            for (subj_label, rel_type, obj_label), batch in grouped.items():
                for i in range(0, len(batch), batch_size):
                    rows = [
                        {
                            "subj":          r["subject"],
                            "obj":           r["object"],
                            "confidence":    r.get("confidence",    0.8),
                            "chunk_ids":     r.get("chunk_ids",     []),
                            "sources":       r.get("sources",       []),
                            "page_numbers":  r.get("page_numbers",  []),
                            "paragraph_ids": r.get("paragraph_ids", []),
                            "stage":         r.get("stage",         "base_graph"),
                            "extracted_at":  ts,
                        }
                        for r in batch[i:i + batch_size]
                    ]
                    cypher = f"""
                    UNWIND $rows AS row
                    MERGE (a:{subj_label} {{name: row.subj}})
                    MERGE (b:{obj_label}  {{name: row.obj}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.confidence    = row.confidence,
                        r.chunk_ids     = row.chunk_ids,
                        r.sources       = row.sources,
                        r.page_numbers  = row.page_numbers,
                        r.paragraph_ids = row.paragraph_ids,
                        r.stage         = row.stage,
                        r.extracted_at  = CASE WHEN r.extracted_at IS NULL
                                            THEN row.extracted_at ELSE r.extracted_at END,
                        r.updated       = timestamp()
                    """
                    s.run(cypher, rows=rows)

        print(f"[neo4j] ✅ Upserted {len(relations)} relations at {ts}")

    # ── Phase 2 hooks (populated by identifier_linker) ───────────────────────

    def set_canonical_ids(
        self,
        entity_name: str,
        label: str,
        ids: Dict[str, str],
    ):
        """
        Attach canonical medical identifiers to an existing node.
        Called by Phase 2 identifier_linker.

        ids dict keys: umls_cui, mesh_id, rxnorm_id, drugbank_id (all optional)
        """
        set_clauses = ", ".join(f"n.{k} = ${k}" for k in ids)
        cypher = f"MATCH (n:{label} {{name: $name}}) SET {set_clauses}"
        with self.driver.session() as s:
            s.run(cypher, {"name": entity_name, **ids})

    def set_relation_temporal(
        self,
        subj: str,
        pred: str,
        obj: str,
        valid_from: Optional[str] = None,
        valid_until: Optional[str] = None,
    ):
        """Attach temporal validity to an existing relationship (Phase 2)."""
        cypher = f"""
        MATCH (a {{name: $subj}})-[r:{pred}]->(b {{name: $obj}})
        SET r.valid_from  = $valid_from,
            r.valid_until = $valid_until
        """
        with self.driver.session() as s:
            s.run(cypher, {
                "subj": subj, "obj": obj,
                "valid_from": valid_from, "valid_until": valid_until,
            })

    # ── Provenance query ──────────────────────────────────────────────────────

    def get_provenance(self, entity_name: str) -> Dict:
        """
        Return full provenance for a named entity: sources, pages, chunk_ids,
        confidence, extraction timestamp, and all its relationships.
        """
        node_cypher = """
        MATCH (n {name: $name})
        RETURN n.name        AS name,
               labels(n)[0]  AS label,
               n.confidence  AS confidence,
               n.sources     AS sources,
               n.pages       AS pages,
               n.chunk_ids   AS chunk_ids,
               n.extracted_at AS extracted_at,
               n.aliases     AS aliases,
               n.umls_cui    AS umls_cui,
               n.mesh_id     AS mesh_id,
               n.rxnorm_id   AS rxnorm_id
        LIMIT 1
        """
        rel_cypher = """
        MATCH (n {name: $name})-[r]->(m)
        RETURN type(r) AS relation, m.name AS target, labels(m)[0] AS target_label,
               r.confidence AS confidence, r.sources AS sources,
               r.page_numbers AS pages, r.extracted_at AS extracted_at
        ORDER BY r.confidence DESC
        """
        with self.driver.session() as s:
            node = s.run(node_cypher, {"name": entity_name}).single()
            rels  = [r.data() for r in s.run(rel_cypher, {"name": entity_name})]
        return {
            "node":          dict(node) if node else {},
            "relationships": rels,
        }

    # ── Query helpers ─────────────────────────────────────────────────────────

    def run_query(self, cypher: str, params: Dict = None) -> List[Dict]:
        with self.driver.session() as s:
            return [r.data() for r in s.run(cypher, params or {})]

    def run_read_query(self, cypher: str, params: Dict = None) -> List[Dict]:
        """Run a query in a READ transaction — Neo4j refuses any write in it."""
        with self.driver.session() as s:
            return s.execute_read(
                lambda tx: [r.data() for r in tx.run(cypher, params or {})]
            )

    def get_node_counts(self) -> Dict[str, int]:
        counts = {}
        with self.driver.session() as s:
            for label in NODE_LABELS:
                cnt = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                counts[label] = cnt
        return counts

    def get_edge_counts(self) -> Dict[str, int]:
        counts = {}
        with self.driver.session() as s:
            for rel in RELATION_TYPES:
                cnt = s.run(
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c"
                ).single()["c"]
                counts[rel] = cnt
        return counts

    def print_summary(self):
        print("\n📊 Graph summary:")
        node_counts = self.get_node_counts()
        for label, cnt in node_counts.items():
            if cnt > 0:
                print(f"  {label:25s}: {cnt:>6,} nodes")
        print()
        edge_counts = self.get_edge_counts()
        for rel, cnt in edge_counts.items():
            if cnt > 0:
                print(f"  {rel:25s}: {cnt:>6,} edges")
        total_nodes = sum(node_counts.values())
        total_edges = sum(edge_counts.values())
        print(f"\n  Total: {total_nodes:,} nodes, {total_edges:,} edges")


    # ── Chunk nodes (provenance graph) ────────────────────────────────────────

    def upsert_chunks_batch(
        self,
        chunks:       List[Dict],
        extracted_at: Optional[str] = None,
        batch_size:   int = 200,
    ):
        """
        Create/update Chunk nodes in Neo4j — the shared document structure
        between graph and vector retrieval.

        Chunk dict keys: chunk_id, text, source, page, chunk_index, token_count

        Each Chunk stores its text, page, and source so provenance queries
        can retrieve the original passage without hitting Qdrant.
        """
        ts = extracted_at or _now_iso()
        with self.driver.session() as s:
            for i in range(0, len(chunks), batch_size):
                rows = [
                    {
                        "chunk_id":    c["chunk_id"],
                        "text":        c.get("text", "")[:2000],  # cap stored text
                        "source":      c.get("source", ""),
                        "page":        c.get("page", 0),
                        "chunk_index": c.get("chunk_index", i),
                        "token_count": c.get("token_count", 0),
                        "extracted_at": ts,
                    }
                    for c in chunks[i:i + batch_size]
                ]
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (c:Chunk {chunk_id: row.chunk_id})
                    SET c.text        = row.text,
                        c.source      = row.source,
                        c.page        = row.page,
                        c.chunk_index = row.chunk_index,
                        c.token_count = row.token_count,
                        c.extracted_at = CASE WHEN c.extracted_at IS NULL
                                           THEN row.extracted_at ELSE c.extracted_at END,
                        c.updated     = timestamp()
                    """,
                    rows=rows,
                )
        print(f"[neo4j] ✅ {len(chunks)} Chunk nodes upserted")

    def link_chunks_to_document(self, sha256: str, chunk_ids: List[str], filename: str = ""):
        """
        Create (:Document {sha256})-[:HAS_CHUNK]->(:Chunk {chunk_id}) edges.

        Document is identified by sha256 (the canonical key set by register_document).
        This avoids creating a second Document node keyed only by filename.
        """
        if not chunk_ids:
            return
        with self.driver.session() as s:
            s.run(
                """
                MERGE (d:Document {sha256: $sha256})
                ON CREATE SET d.filename = $filename
                WITH d
                UNWIND $chunk_ids AS cid
                MATCH (c:Chunk {chunk_id: cid})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                {"sha256": sha256, "filename": filename, "chunk_ids": chunk_ids},
            )
        print(f"[neo4j] ✅ Document sha256={sha256[:10]}… → {len(chunk_ids)} HAS_CHUNK edges")

    def link_chunks_to_entities(
        self,
        chunk_entity_conf: Dict[str, Dict[str, float]],
    ) -> int:
        """
        Create (:Chunk)-[:MENTIONS {confidence}]->(:Entity) edges.

        Args:
            chunk_entity_conf: {chunk_id → {entity_name → confidence}}
                               Built by graph_loader._build_chunk_entity_confidence()

        Returns:
            Total number of MENTIONS edges created/updated
        """
        total = 0
        with self.driver.session() as s:
            for chunk_id, entities in chunk_entity_conf.items():
                rows = [
                    {"name": name, "confidence": conf}
                    for name, conf in entities.items()
                    if name
                ]
                if not rows:
                    continue
                result = s.run(
                    """
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    UNWIND $rows AS row
                    MATCH (e {name: row.name})
                    MERGE (c)-[r:MENTIONS]->(e)
                    SET r.confidence = row.confidence,
                        r.updated    = timestamp()
                    RETURN count(r) AS cnt
                    """,
                    {"chunk_id": chunk_id, "rows": rows},
                )
                rec = result.single()
                total += rec["cnt"] if rec else 0
        return total

    # ── Chunk provenance query ────────────────────────────────────────────────

    def get_chunk_provenance(self, entity_name: str) -> List[Dict]:
        """
        Return the chunks that mention a named entity, with page and source.
        Used by the query pipeline to trace answers back to source passages.
        """
        cypher = """
        MATCH (c:Chunk)-[r:MENTIONS]->(e {name: $name})
        RETURN c.chunk_id   AS chunk_id,
               c.source     AS source,
               c.page       AS page,
               c.text       AS text_preview,
               r.confidence AS mention_confidence
        ORDER BY r.confidence DESC
        LIMIT 10
        """
        return self.run_query(cypher, {"name": entity_name})


if __name__ == "__main__":
    with Neo4jClient() as c:
        print(c.verify_connection())
        c.create_indexes()
        c.print_summary()
