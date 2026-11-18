"""GET /graph/explore: every edge must end on a node the client receives."""

import pytest
from fastapi.testclient import TestClient

import backend.main as main


def _node(name, label="Drug"):
    return {"name": name, "label": label, "confidence": 0.9, "stage": "base_graph",
            "pages": [1], "sources": ["guide"], "umls_cui": None, "mesh_id": None,
            "rxnorm_id": None}


def _edge(source, target, relation="TREATS"):
    return {"source": source, "relation": relation, "target": target, "confidence": 0.9,
            "stage": "base_graph", "pages": [1], "sources": ["guide"]}


class FakeNeo4j:
    """Answers the explore queries in order: nodes, edges, then neighbour nodes."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def run_query(self, cypher, params=None):
        self.calls.append((cypher, params))
        return self.answers.pop(0)


@pytest.fixture
def explore(monkeypatch):
    def run(fake, **query):
        monkeypatch.setattr(main, "get_neo4j", lambda: fake)
        r = TestClient(main.app).get("/graph/explore", params=query)
        assert r.status_code == 200
        return r.json()
    return run


def test_unfiltered_view_drops_edges_to_nodes_outside_the_page(explore):
    fake = FakeNeo4j([_node("Metformin"), _node("Type 2 Diabetes", "Disease")],
                     [_edge("Metformin", "Type 2 Diabetes"), _edge("Lisinopril", "Hypertension")])
    body = explore(fake, limit=2)

    assert body["edges"] == [_edge("Metformin", "Type 2 Diabetes")]
    assert len(fake.calls) == 2                     # no neighbour lookup without a label
    assert fake.calls[0][1]["labels"] == main.NODE_LABELS   # Chunk/Document nodes excluded


def test_label_filter_adds_neighbour_nodes_so_edges_can_be_drawn(explore):
    fake = FakeNeo4j([_node("Metformin")],
                     [_edge("Metformin", "Type 2 Diabetes"), _edge("Metformin", "Lactic Acidosis", "CAUSES")],
                     [_node("Type 2 Diabetes", "Disease"), _node("Lactic Acidosis", "SideEffect")])
    body = explore(fake, label="Drug")

    assert [n["name"] for n in body["nodes"]] == ["Metformin", "Type 2 Diabetes", "Lactic Acidosis"]
    assert len(body["edges"]) == 2
    assert fake.calls[2][1] == {"names": ["Type 2 Diabetes", "Lactic Acidosis"]}


def test_one_node_per_name(explore):
    fake = FakeNeo4j([_node("Metformin")], [_edge("Metformin", "Insulin")],
                     [_node("Insulin", "Drug"), _node("Insulin", "Protein")])
    body = explore(fake, label="Drug")

    assert [(n["name"], n["label"]) for n in body["nodes"]] == [("Metformin", "Drug"), ("Insulin", "Drug")]


def test_label_is_passed_as_a_parameter_not_spliced_into_cypher(explore):
    fake = FakeNeo4j([], [])
    explore(fake, label="Disease")
    assert "Disease" not in fake.calls[0][0]
    assert fake.calls[0][1]["labels"] == ["Disease"]
