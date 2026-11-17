"""pipeline/graph_retriever.py — how graph facts become cited context."""

from pipeline.graph_retriever import (
    RetrievedSubgraph,
    SubgraphEdge,
    _rows_to_subgraph,
    format_subgraph_for_context,
)


def _row(**overrides):
    row = {
        "source": "Metformin", "source_label": "Drug", "source_conf": 0.95,
        "source_pages": [1], "source_umls": "C0025598", "source_mesh": "",
        "relation": "CAUSES", "rel_conf": 0.9, "sources": ["diabetes.pdf"],
        "pages": [2, 3], "stage": "base_graph",
        "target": "Lactic Acidosis", "target_label": "SideEffect", "target_conf": 0.9,
        "doc_source": "diabetes_management_guide.pdf", "doc_page": 2,
        "excerpt": "Metformin can rarely cause lactic acidosis",
    }
    row.update(overrides)
    return row


def test_provenance_rows_get_file_and_page_citation():
    sg = _rows_to_subgraph([_row()], ["Metformin"], include_provenance=True)

    assert sg.path_texts == [
        "Metformin --[CAUSES]--> Lactic Acidosis [diabetes_management_guide.pdf, p.2]"
    ]
    edge = sg.edges[0]
    assert (edge.doc_source, edge.doc_page) == ("diabetes_management_guide.pdf", 2)
    assert edge.excerpt.startswith("Metformin can rarely")
    assert {n.name: n.label for n in sg.nodes} == {"Metformin": "Drug", "Lactic Acidosis": "SideEffect"}
    assert sg.nodes[0].umls_cui == "C0025598"


def test_file_without_page_cites_file_only():
    sg = _rows_to_subgraph([_row(doc_page=0)], [], include_provenance=True)
    assert sg.path_texts[0].endswith("[diabetes_management_guide.pdf]")


def test_without_provenance_falls_back_to_relation_pages():
    sg = _rows_to_subgraph([_row()], [], include_provenance=False)

    assert sg.path_texts == ["Metformin --[CAUSES]--> Lactic Acidosis [p.2,3]"]
    assert sg.edges[0].doc_source == "" and sg.edges[0].excerpt == ""


def test_incomplete_rows_are_skipped():
    rows = [_row(target=None), _row(relation=""), _row(source="")]
    assert _rows_to_subgraph(rows, [], include_provenance=True).is_empty()


def test_merge_duplicate_edges_keeps_highest_confidence_and_rebuilds_citations():
    # The intent-aware query returns one row per Evidence node, so the same
    # fact from the same page can come back several times.
    sg = _rows_to_subgraph(
        [_row(rel_conf=0.6), _row(rel_conf=0.92), _row(doc_page=5, rel_conf=0.7)],
        [], include_provenance=True,
    )
    sg.merge_duplicate_edges()

    assert sorted(e.confidence for e in sg.edges) == [0.7, 0.92]
    assert sorted(sg.path_texts) == [
        "Metformin --[CAUSES]--> Lactic Acidosis [diabetes_management_guide.pdf, p.2]",
        "Metformin --[CAUSES]--> Lactic Acidosis [diabetes_management_guide.pdf, p.5]",
    ]


def test_context_block_includes_citations_excerpts_and_anchors():
    sg = _rows_to_subgraph([_row(excerpt="E" * 300)], [], include_provenance=True)
    sg.intent_aware = True
    sg.evidence_chunks = ["diabetes_management_guide.pdf:2"]

    text = format_subgraph_for_context(sg)

    assert text.splitlines()[0] == "[KNOWLEDGE SUBGRAPH (INTENT-AWARE) — 2 nodes, 1 edges]"
    assert "Metformin (Drug [UMLS:C0025598])" in text
    assert ("Metformin --[CAUSES]--> Lactic Acidosis  "
            "(conf=0.90, diabetes_management_guide.pdf, p.2)") in text
    assert 'excerpt: "' + "E" * 120 + '"' in text         # excerpt capped at 120 chars
    assert "E" * 121 not in text
    assert "Evidence anchors: 1 source(s) — diabetes_management_guide.pdf:2" in text


def test_context_block_falls_back_to_page_numbers():
    sg = RetrievedSubgraph(edges=[SubgraphEdge("A", "TREATS", "B", confidence=0.8, pages=[4, 5, 6, 7])])
    text = format_subgraph_for_context(sg)
    assert "INDUCED-SUBGRAPH" in text
    assert "A --[TREATS]--> B  (conf=0.80, p.4,5,6)" in text


def test_empty_subgraph_gives_empty_context():
    assert format_subgraph_for_context(RetrievedSubgraph()) == ""
