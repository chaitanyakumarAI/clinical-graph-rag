"""POST /query end to end, with every LLM / Neo4j / Qdrant call faked."""

import pytest
from fastapi.testclient import TestClient

import backend.main as main
from llm import RateLimitExhausted
from pipeline.completeness import CoverageReport
from pipeline.graph_retriever import RetrievedSubgraph, _rows_to_subgraph
from pipeline.query_planner import QueryPlan
from pipeline.query_rewriter import RewrittenQuery

TMP = "tmpk3j9x"   # API uploads are parsed from a temp file with this kind of name


def _chunk(cid, source, page, score, text=None):
    return {"chunk_id": cid, "source": source, "page": page, "score": score,
            "text": text or f"text of {cid} about metformin and lactic acidosis"}


def _edge_row(target, conf, page=2):
    return {
        "source": "Metformin", "source_label": "Drug", "relation": "CAUSES",
        "target": target, "target_label": "SideEffect", "rel_conf": conf,
        "pages": [page], "doc_source": TMP, "doc_page": page,
        "excerpt": f"Metformin may cause {target.lower()}",
    }


class Recorder:
    def __init__(self, result=None):
        self.result, self.calls = result, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result(*args, **kwargs) if callable(self.result) else self.result


@pytest.fixture
def app(monkeypatch):
    """Patch the pipeline seams in backend.main; tests override what they need."""
    fakes = {
        "rewrite": Recorder(lambda q, provider: RewrittenQuery(original=q, rewritten=q)),
        "vector": Recorder(lambda q, top_k: [
            _chunk("c1", TMP, 2, 0.81), _chunk("c2", TMP, 2, 0.62), _chunk("c3", "other", 7, 0.5),
        ]),
        "plan": Recorder(QueryPlan(route="vector_only", reason="definition")),
        "link": Recorder([{"query_term": "metformin", "neo4j_name": "Metformin", "label": "Drug"}]),
        "subgraph": Recorder((RetrievedSubgraph(), [])),   # what the real one returns when empty
        "answer": Recorder("Metformin can cause lactic acidosis [diabetes.pdf, p.2]."),
        "coverage": Recorder(CoverageReport()),          # nothing missing by default
    }
    monkeypatch.setattr(main, "check_coverage", fakes["coverage"])
    monkeypatch.setattr(main, "get_neo4j", lambda: object())
    monkeypatch.setattr(main, "rewrite_query", fakes["rewrite"])
    monkeypatch.setattr(main, "retrieve_vector_context", fakes["vector"])
    monkeypatch.setattr(main, "plan_query", fakes["plan"])
    monkeypatch.setattr(main, "link_entities", fakes["link"])
    monkeypatch.setattr(main, "retrieve_subgraph", fakes["subgraph"])
    monkeypatch.setattr(main, "generate_answer", fakes["answer"])
    monkeypatch.setattr("graph.document_registry.get_source_map", lambda client: {TMP: "diabetes.pdf"})
    monkeypatch.setattr("pipeline.entity_extractor_lite.spot_entities_in_chunks",
                        lambda chunks, provider: ["Lactic Acidosis"])
    monkeypatch.setattr("pipeline.reranker._score_candidates", lambda q, texts: [0.5] * len(texts))
    return TestClient(main.app), fakes


def test_vector_only_skips_graph_and_dedupes_document_citations(app):
    client, fakes = app

    r = client.post("/query", json={"question": "What is metformin?", "force_no_gds": True})

    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "vector_only"
    assert fakes["subgraph"].calls == [] and fakes["link"].calls == []
    # temp-file source mapped back to the uploaded filename
    assert body["vector_sources"] == ["diabetes.pdf p.2", "other p.7"]
    # two chunks from the same (file, page) collapse into one citation, best score kept
    docs = [p for p in body["provenance"] if p["type"] == "document"]
    assert [(p["source"], p["page"]) for p in docs] == [("diabetes.pdf", 2), ("other", 7)]
    assert docs[0]["score"] == pytest.approx(0.4 * 0.81 + 0.6 * 0.5, abs=1e-3)
    # the answer LLM gets the original question and exactly the reported contexts
    (question, context), _ = fakes["answer"].calls[0]
    assert question == "What is metformin?"
    assert context == "\n\n".join(body["retrieved_contexts"])


def test_hybrid_route_adds_cited_graph_facts_and_filters_low_confidence(app):
    client, fakes = app
    fakes["plan"].result = QueryPlan(route="hybrid_single", reason="one hop",
                                     entities_mentioned=["Metformin"],
                                     target_relations=["CAUSES"], target_labels=["SideEffect"])
    subgraph = _rows_to_subgraph([_edge_row("Lactic Acidosis", 0.9), _edge_row("Rash", 0.3)],
                                 ["Metformin"], include_provenance=True)
    subgraph.intent_aware = True
    subgraph.evidence_chunks = [f"{TMP}:2"]
    fakes["subgraph"].result = (subgraph, [])

    body = client.post("/query", json={"question": "metformin side effects",
                                       "force_no_gds": True, "min_confidence": 0.5}).json()

    # planner entities + entities spotted in the chunks are both linked
    (names, _client), _ = fakes["link"].calls[0]
    assert sorted(names) == ["Lactic Acidosis", "Metformin"]
    # planner intent reaches graph retrieval
    assert fakes["subgraph"].calls[0][1]["target_relations"] == ["CAUSES"]
    # low-confidence edge dropped from the reranked candidates; temp name mapped
    assert body["graph_paths"] == ["Metformin --[CAUSES]--> Lactic Acidosis [diabetes.pdf, p.2]"]
    graph_prov = [p for p in body["provenance"] if p["type"] == "graph"]
    assert graph_prov[0]["doc_source"] == "diabetes.pdf"
    assert graph_prov[0]["excerpt"] == "Metformin may cause lactic acidosis"
    assert body["subgraph"]["evidence_chunks"] == ["diabetes.pdf:2"]
    assert body["intent_aware"] is True
    # structured subgraph block is the first context the answer LLM sees
    assert body["retrieved_contexts"][0].startswith("[KNOWLEDGE SUBGRAPH (INTENT-AWARE)")


def test_each_sub_query_gets_its_own_vector_search(app):
    client, fakes = app
    fakes["rewrite"].result = lambda q, provider: RewrittenQuery(
        original=q, rewritten="Diabetes drugs that interact with blood pressure drugs",
        sub_queries=["diabetes drug interactions", "blood pressure drug interactions"])
    per_query = {
        "Diabetes drugs that interact with blood pressure drugs": [_chunk("a1", "d.pdf", 1, .9), _chunk("a2", "d.pdf", 2, .8)],
        "diabetes drug interactions": [_chunk("a1", "d.pdf", 1, .9), _chunk("b1", "i.pdf", 1, .7)],
        "blood pressure drug interactions": [_chunk("c1", "h.pdf", 1, .6)],
    }
    fakes["vector"].result = lambda q, top_k: per_query[q]

    body = client.post("/query", json={"question": "which T2DM drugs interact with BP meds"}).json()

    assert [c[0][0] for c in fakes["vector"].calls] == list(per_query)
    assert body["sub_queries"] == ["diabetes drug interactions", "blood pressure drug interactions"]
    # round-robin merge: rank 1 of every query (a1, a1-duplicate skipped, c1),
    # then rank 2 of every query (a2, b1)
    assert body["vector_sources"] == ["d.pdf p.1", "h.pdf p.1", "d.pdf p.2", "i.pdf p.1"]


def test_rewrite_can_be_turned_off(app):
    client, fakes = app
    client.post("/query", json={"question": "AF symptoms", "use_query_rewrite": False})
    assert fakes["rewrite"].calls == []
    assert fakes["vector"].calls[0][0][0] == "AF symptoms"


def test_exhausted_quota_becomes_429(app):
    client, fakes = app

    def quota_gone(*args, **kwargs):
        raise RateLimitExhausted("groq", 1800, "tokens per day")
    fakes["plan"].result = quota_gone

    r = client.post("/query", json={"question": "q"})

    assert r.status_code == 429
    assert r.json()["retry_after_seconds"] == 1800
    assert "groq" in r.json()["detail"]


def _raise(exc):
    def raiser(*args, **kwargs):
        raise exc
    return raiser


def test_qdrant_down_becomes_503(app):
    from vector.qdrant_client import VectorStoreUnavailable
    client, fakes = app
    fakes["vector"].result = _raise(VectorStoreUnavailable("Cannot reach Qdrant"))

    r = client.post("/query", json={"question": "q"})

    assert r.status_code == 503
    assert r.json()["service"] == "Qdrant"
    assert "QDRANT_URL" in r.json()["detail"]


def test_neo4j_down_becomes_503(app):
    import neo4j.exceptions
    client, fakes = app
    fakes["plan"].result = QueryPlan(route="hybrid_single", reason="r", entities_mentioned=["Metformin"])
    fakes["link"].result = _raise(neo4j.exceptions.ServiceUnavailable("Couldn't connect to localhost:7687"))

    r = client.post("/query", json={"question": "q"})

    assert r.status_code == 503
    assert r.json()["service"] == "Neo4j"


def test_qdrant_transport_error_is_wrapped():
    pytest.importorskip("qdrant_client")
    from qdrant_client.http.exceptions import ResponseHandlingException
    from vector import qdrant_client as store_module

    store = store_module.QdrantStore.__new__(store_module.QdrantStore)
    store.collection = "medical_chunks"
    store.client = type("C", (), {"query_points": _raise(ResponseHandlingException(ConnectionError("refused")))})()

    with pytest.raises(store_module.VectorStoreUnavailable, match="Cannot reach Qdrant"):
        store.search([0.0] * 3)


# ── Completeness check (step 9) ───────────────────────────────────────────────

def _multi_part(fakes):
    fakes["plan"].result = QueryPlan(route="hybrid_single", reason="r")
    fakes["coverage"].result = CoverageReport(parts=[
        {"part": "diabetes drugs that interact with blood pressure drugs", "covered": True},
        {"part": "effect of the interaction", "covered": False},
    ])
    first = [_chunk("c1", TMP, 2, 0.81)]
    repair = [_chunk("c1", TMP, 2, 0.81),                                  # already in context
              _chunk("r1", "interactions.pdf", 1, 0.77, text="ACE inhibitors raise hypoglycemia risk"),
              _chunk("r2", "interactions.pdf", 2, 0.40, text="unrelated passage")]
    fakes["vector"].result = lambda q, top_k: repair if q == "effect of the interaction" else first
    fakes["answer"].result = lambda q, ctx, provider, required_parts=None: (
        "complete answer" if required_parts else "draft answer")


def test_missing_part_triggers_targeted_search_and_one_regeneration(app):
    client, fakes = app
    _multi_part(fakes)

    body = client.post("/query", json={"question": "diabetes drugs vs BP drugs: which interact, what happens?",
                                       "force_no_gds": True}).json()

    assert body["answer"] == "complete answer"
    assert len(fakes["answer"].calls) == 2                               # draft + one repair
    assert fakes["answer"].calls[1][1]["required_parts"] == [
        "diabetes drugs that interact with blood pressure drugs", "effect of the interaction"]
    # the targeted search ran for the missing part only
    assert [c[0][0] for c in fakes["vector"].calls][-1] == "effect of the interaction"
    # new chunks (not the duplicate) reach the context, provenance and response
    assert any("ACE inhibitors raise hypoglycemia risk" in c for c in body["retrieved_contexts"])
    assert sum("text of c1" in c for c in body["retrieved_contexts"]) == 1
    assert ("interactions.pdf", 1) in {(p["source"], p["page"]) for p in body["provenance"]
                                       if p["type"] == "document"}
    (_, regenerated_context), _ = fakes["answer"].calls[1]
    assert regenerated_context == "\n\n".join(body["retrieved_contexts"])
    assert body["completeness"]["missing"] == ["effect of the interaction"]
    assert body["completeness"]["repaired"] is True
    assert "interactions.pdf p.1" in body["completeness"]["added_sources"]


def test_fully_covered_answer_is_not_regenerated(app):
    client, fakes = app
    fakes["plan"].result = QueryPlan(route="hybrid_single", reason="r")
    fakes["coverage"].result = CoverageReport(parts=[{"part": "side effects of metformin", "covered": True}])

    body = client.post("/query", json={"question": "metformin side effects", "force_no_gds": True}).json()

    assert len(fakes["answer"].calls) == 1
    assert body["completeness"] == {"parts": ["side effects of metformin"], "missing": [],
                                    "repaired": False, "added_sources": []}


def test_single_part_definition_question_skips_the_check(app):
    client, fakes = app   # route vector_only, no sub-queries
    body = client.post("/query", json={"question": "What is metformin?"}).json()
    assert fakes["coverage"].calls == []
    assert body["completeness"] is None


def test_completeness_check_can_be_turned_off(app):
    client, fakes = app
    _multi_part(fakes)
    body = client.post("/query", json={"question": "q", "use_completeness_check": False,
                                       "force_no_gds": True}).json()
    assert fakes["coverage"].calls == [] and len(fakes["answer"].calls) == 1
    assert body["answer"] == "draft answer"


def test_merge_round_robin_interleaves_and_dedupes():
    lists = [
        [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}],
        [{"chunk_id": "b"}, {"chunk_id": "d"}],
        [],
    ]
    assert [r["chunk_id"] for r in main._merge_round_robin(lists)] == ["a", "b", "d", "c"]
    assert main._merge_round_robin([]) == []
