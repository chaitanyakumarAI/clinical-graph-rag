"""
Cross-document provenance: a second PDF must add to shared entities/edges,
not overwrite them, and deleting a PDF must take its provenance back out.

The first tests capture the Cypher sent to a fake driver (no database). The
integration test runs against a real Neo4j when NEO4J_TEST_URI is set, e.g.
    NEO4J_TEST_URI=bolt://localhost:7687 NEO4J_TEST_PASSWORD=... pytest -k neo4j
It only touches nodes it creates (unique name/id prefix) and removes them.
"""

from __future__ import annotations
import os
import re
import uuid

import pytest

from graph import neo4j_client
from graph.neo4j_client import Neo4jClient

LIST_PROPS = ["aliases", "sources", "chunk_ids", "pages", "page_numbers", "paragraph_ids"]


class _RecordingSession:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, *args, **kwargs):
        self.log.append(cypher)


def _client_recording_cypher():
    client = Neo4jClient.__new__(Neo4jClient)   # skip connecting
    log = []
    client.driver = type("Driver", (), {"session": lambda self, **kw: _RecordingSession(log)})()
    return client, log


def _assignments(cypher):
    """{property: expression} for every `x.prop = expr` assignment in a SET clause."""
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"\b[nr]\.(\w+)\s*=\s*(.+?)(?:,\n|\n|$)", cypher)}


ENTITY = {"type": "Drug", "name": "Metformin", "aliases": ["Glucophage"], "sources": ["a"],
          "confidence": 0.9, "chunk_ids": ["c1"], "pages": [1]}
RELATION = {"subject_type": "Drug", "subject": "Metformin", "predicate": "TREATS",
            "object_type": "Disease", "object": "Type 2 Diabetes", "confidence": 0.9,
            "chunk_ids": ["c1"], "sources": ["a"], "page_numbers": [1], "paragraph_ids": [0]}


@pytest.mark.parametrize("write", [
    lambda c: c.upsert_entities_batch([ENTITY]),
    lambda c: c.upsert_entity(ENTITY),
    lambda c: c.upsert_relations_batch([RELATION]),
    lambda c: c.upsert_relation(RELATION),
], ids=["entities_batch", "entity", "relations_batch", "relation"])
def test_upserts_union_lists_and_keep_max_confidence(write):
    client, log = _client_recording_cypher()
    write(client)

    sets = _assignments(log[-1])
    written = [p for p in LIST_PROPS if p in sets]
    assert written, "expected list properties in the SET clause"
    for prop in written:
        assert sets[prop].startswith("reduce(acc = []"), f"{prop} is overwritten: {sets[prop]}"
    assert sets["confidence"].startswith("CASE WHEN")
    assert "apoc." not in log[-1]          # must work without the APOC plugin
    assert "/ 2.0" not in log[-1]          # no order-dependent running average


# ── Integration test against a real Neo4j ─────────────────────────────────────

NEO4J_TEST_URI = os.getenv("NEO4J_TEST_URI")
neo4j_only = pytest.mark.skipif(not NEO4J_TEST_URI, reason="set NEO4J_TEST_URI to run")


@pytest.fixture
def live_client(monkeypatch):
    monkeypatch.setattr(neo4j_client, "NEO4J_URI", NEO4J_TEST_URI)
    monkeypatch.setattr(neo4j_client, "NEO4J_USER", os.getenv("NEO4J_TEST_USER", "neo4j"))
    monkeypatch.setattr(neo4j_client, "NEO4J_PASSWORD", os.getenv("NEO4J_TEST_PASSWORD", "password"))
    client = Neo4jClient()
    tag = "Prov" + uuid.uuid4().hex[:8]       # unique per run, survives title-casing
    yield client, tag
    client.run_query(
        """
        MATCH (n)
        WHERE n.name STARTS WITH $tag OR n.chunk_id STARTS WITH $tag
           OR n.sha256 STARTS WITH $tag OR n.subject STARTS WITH $tag
        DETACH DELETE n
        """,
        {"tag": tag},
    )
    client.close()


def _ingest(client, tag, doc, page, entity_conf, relation_conf, extra_drug=None):
    """Load one tiny 'PDF' through the real dedup + graph-loader path."""
    from graph.graph_loader import load_to_graph
    from ingestion.deduplicator import deduplicate
    from ingestion.entity_extractor import ExtractedEntity, ExtractedRelation, ExtractionResult

    metformin, diabetes = f"{tag} Metformin", f"{tag} Diabetes"
    chunk = {"chunk_id": f"{tag}-{doc}-c1", "source": f"{tag}-tmp-{doc}", "page": page,
             "chunk_index": 0, "token_count": 20,
             "text": f"{metformin} treats {diabetes}." + (f" It interacts with {extra_drug}." if extra_drug else "")}
    entities = [ExtractedEntity(type="Drug", name=metformin, confidence=entity_conf),
                ExtractedEntity(type="Disease", name=diabetes, confidence=0.9)]
    relations = [ExtractedRelation(subject_type="Drug", subject=metformin, predicate="TREATS",
                                   object_type="Disease", object=diabetes, confidence=relation_conf)]
    if extra_drug:
        entities.append(ExtractedEntity(type="Drug", name=extra_drug, confidence=0.9))
        relations.append(ExtractedRelation(subject_type="Drug", subject=metformin,
                                           predicate="INTERACTS_WITH", object_type="Drug",
                                           object=extra_drug, confidence=0.9))
    result = ExtractionResult(entities=entities, relations=relations, chunk_id=chunk["chunk_id"],
                              source=chunk["source"], page=page)
    ents, rels = deduplicate([result])
    load_to_graph([chunk], ents, rels, [result], client,
                  filename=f"{doc}.pdf", sha256=f"{tag}-sha-{doc}")
    return metformin, diabetes


def _entity(client, name):
    return client.run_query("MATCH (n:Drug {name: $n}) RETURN n.chunk_ids AS chunk_ids, "
                            "n.sources AS sources, n.pages AS pages, n.confidence AS conf",
                            {"n": name})[0]


def _treats(client, name):
    return client.run_query("MATCH (:Drug {name: $n})-[r:TREATS]->() RETURN r.chunk_ids AS chunk_ids, "
                            "r.page_numbers AS pages, r.confidence AS conf", {"n": name})[0]


def _evidence_chunks(client, name):
    rows = client.run_query("MATCH (:Drug {name: $n})-[:HAS_EVIDENCE]->(ev)-[:FROM_CHUNK]->(c) "
                            "RETURN c.chunk_id AS cid", {"n": name})
    return sorted(r["cid"] for r in rows)


@neo4j_only
def test_neo4j_second_pdf_adds_provenance_and_delete_removes_it(live_client):
    from graph.document_registry import delete_document

    client, tag = live_client
    metformin, _ = _ingest(client, tag, "A", page=1, entity_conf=0.7, relation_conf=0.7)
    _ingest(client, tag, "B", page=3, entity_conf=0.95, relation_conf=0.9,
            extra_drug=f"{tag} Lisinopril")

    a, b = f"{tag}-A-c1", f"{tag}-B-c1"
    ent = _entity(client, metformin)
    assert sorted(ent["chunk_ids"]) == [a, b]                       # was [b] before the fix
    assert sorted(ent["sources"]) == [f"{tag}-tmp-A", f"{tag}-tmp-B"]
    assert sorted(ent["pages"]) == [1, 3]
    assert ent["conf"] == pytest.approx(0.95)
    edge = _treats(client, metformin)
    assert sorted(edge["chunk_ids"]) == [a, b]
    assert sorted(edge["pages"]) == [1, 3]
    assert edge["conf"] == pytest.approx(0.9)

    # Re-ingesting the same PDF (CLI path) changes nothing.
    _ingest(client, tag, "B", page=3, entity_conf=0.95, relation_conf=0.9,
            extra_drug=f"{tag} Lisinopril")
    assert sorted(_entity(client, metformin)["chunk_ids"]) == [a, b]
    assert _treats(client, metformin)["conf"] == pytest.approx(0.9)

    # Deleting PDF A removes its chunk, evidence and provenance; B's survives.
    # Evidence: TREATS from A and from B, INTERACTS_WITH from B.
    assert _evidence_chunks(client, metformin) == sorted([a, b, b])
    result = delete_document(f"{tag}-sha-A", client)
    assert result["doc_deleted"] and result["chunks_deleted"] == 1
    ent = _entity(client, metformin)
    assert ent["chunk_ids"] == [b]
    assert ent["sources"] == [f"{tag}-tmp-B"]
    assert _treats(client, metformin)["chunk_ids"] == [b]
    assert a not in _evidence_chunks(client, metformin)
    assert b in _evidence_chunks(client, metformin)
