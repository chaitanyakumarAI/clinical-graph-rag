"""pipeline/cypher_generator.py — the read-only guard on LLM-generated Cypher."""

import pytest

from pipeline import cypher_generator
from pipeline.cypher_generator import (
    _inject_limit_if_missing,
    _is_safe_cypher,
    execute_first_nonempty,
    generate_cypher,
)

SAFE = [
    "MATCH (a:Drug {name: $entity1})-[r:TREATS]->(b) WHERE r.confidence > 0.5 "
    "RETURN a.name AS source, type(r) AS relation, b.name AS target LIMIT $limit",
    "OPTIONAL MATCH (a {name: $entity1})-[r]->(b) RETURN a, r, b LIMIT 10",
    "WITH $entity1 AS n MATCH (a {name: n})-[*1..3]-(b) RETURN b.name LIMIT 25",
    "  match (a)-[r]->(b) return a.name limit 5  ",
]

UNSAFE = [
    "CREATE (n:Drug {name: 'x'}) RETURN n LIMIT 1",                       # must start with MATCH/WITH
    "MATCH (n) DETACH DELETE n LIMIT 1",
    "MATCH (n)\nDETACH\nDELETE n LIMIT 1",                                # newline-separated
    "MATCH (n) SET\tn.name = 'x' RETURN n LIMIT 1",                       # tab-separated
    "MATCH (n) WITH n MERGE (m:Drug {name: 'y'}) RETURN m LIMIT 1",
    "MATCH (n) REMOVE n.name RETURN n LIMIT 1",
    "MATCH (n) CALL apoc.periodic.iterate('MATCH (x) RETURN x', 'DELETE x', {}) RETURN n LIMIT 1",
    "MATCH (n) CALL dbms.security.createUser('u', 'p') RETURN n LIMIT 1",
    "WITH 1 AS x LOAD CSV FROM 'http://evil' AS row RETURN row LIMIT 1",
    "MATCH (n) FOREACH (x IN [1] | SET n.p = x) RETURN n LIMIT 1",
    "MATCH (n) drop index foo LIMIT 1",                                    # case-insensitive
    "SHOW USERS",
    "RETURN 1 LIMIT 1",
]


@pytest.mark.parametrize("query", SAFE)
def test_read_queries_pass(query):
    assert _is_safe_cypher(query)


@pytest.mark.parametrize("query", UNSAFE)
def test_write_or_procedure_queries_are_rejected(query):
    assert not _is_safe_cypher(query)


def test_keyword_inside_identifier_is_not_a_false_positive():
    # whole-word matching: "settings", "created_at" and "offset" contain keywords
    q = "MATCH (n) WHERE n.settings IS NOT NULL RETURN n.created_at, n.offset LIMIT 5"
    assert _is_safe_cypher(q)


def test_query_without_limit_is_rejected_until_one_is_injected():
    q = "MATCH (n) RETURN n"
    assert not _is_safe_cypher(q)
    fixed = _inject_limit_if_missing(q, 25)
    assert fixed == "MATCH (n) RETURN n LIMIT 25"
    assert _is_safe_cypher(fixed)
    assert _inject_limit_if_missing(fixed, 99) == fixed   # existing LIMIT kept


def test_generate_cypher_keeps_only_safe_string_candidates(monkeypatch, fake_llm):
    monkeypatch.setattr(cypher_generator, "chat_json", fake_llm({"queries": [
        "MATCH (a {name: $entity1})-[r:CAUSES]->(b) RETURN a, r, b",   # LIMIT gets injected
        "MATCH (n) DETACH DELETE n",
        None,
        {"not": "a string"},
        "MATCH (a)-[r]->(b) RETURN a LIMIT $limit",
    ]}))
    monkeypatch.setattr("graph.schema_inspector.format_schema_for_prompt", lambda schema: "schema")

    queries = generate_cypher(
        "side effects of metformin?",
        linked_entities=[{"query_term": "metformin", "neo4j_name": "Metformin", "label": "Drug"}],
        schema={}, limit=30,
    )

    assert queries == [
        "MATCH (a {name: $entity1})-[r:CAUSES]->(b) RETURN a, r, b LIMIT 30",
        "MATCH (a)-[r]->(b) RETURN a LIMIT $limit",
    ]


def test_generate_cypher_returns_empty_on_llm_failure(monkeypatch, fake_llm):
    monkeypatch.setattr(cypher_generator, "chat_json", fake_llm(RuntimeError("down")))
    monkeypatch.setattr("graph.schema_inspector.format_schema_for_prompt", lambda schema: "schema")
    assert generate_cypher("q", linked_entities=[], schema={}) == []


class _FakeNeo4j:
    """Records which method ran each query; only run_read_query is allowed."""

    def __init__(self, results):
        self.results = list(results)
        self.read_calls = []

    def run_read_query(self, cypher, params=None):
        self.read_calls.append((cypher, params))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def run_query(self, *args, **kwargs):  # pragma: no cover - must never be used
        raise AssertionError("generated Cypher must run in a READ transaction")


def test_execute_first_nonempty_uses_read_transactions_and_skips_failures():
    client = _FakeNeo4j([RuntimeError("syntax error"), [], [{"source": "A"}], [{"never": "reached"}]])

    rows = execute_first_nonempty(["q1", "q2", "q3", "q4"], client, {"limit": 5})

    assert rows == [{"source": "A"}]
    assert [c[0] for c in client.read_calls] == ["q1", "q2", "q3"]


def test_execute_first_nonempty_all_empty():
    assert execute_first_nonempty(["q1"], _FakeNeo4j([[]]), {}) == []
