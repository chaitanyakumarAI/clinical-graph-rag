"""
pipeline/graph_retriever.py
Intent-aware graph retrieval — executes targeted Neo4j queries driven by the
QueryPlan's target_relations / target_labels instead of blind subgraph expansion.

Strategy
--------
When target_relations is non-empty (hybrid_single / hybrid_multi with a clear
intent), execute a TARGETED Cypher that traverses only the requested relation
types, joins Evidence nodes in the same query, and returns per-edge provenance
(doc_source, doc_page, excerpt).

When target_relations is empty (gds_augmented, vector_only fallback, or planner
uncertainty), fall back to the classic induced-subgraph expansion so multi-hop
topology queries still work.

Public API (backward-compatible):
    retrieve_graph_context(...)  → (List[Dict], List[str])
    retrieve_subgraph(...)       → (RetrievedSubgraph, List[str])

    format_for_context(...)           → str   (legacy)
    format_subgraph_for_context(...)  → str   (structured, with citations)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.neo4j_client import Neo4jClient

_CONFIDENCE_THRESHOLD = 0.5


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SubgraphNode:
    name:       str
    label:      str
    confidence: float = 0.8
    pages:      List[int] = field(default_factory=list)
    umls_cui:   str = ""
    mesh_id:    str = ""


@dataclass
class SubgraphEdge:
    source:       str
    relation:     str
    target:       str
    confidence:   float = 0.8
    pages:        List[int] = field(default_factory=list)
    sources:      List[str] = field(default_factory=list)
    stage:        str = "base_graph"
    evidence_ids: List[str] = field(default_factory=list)
    # Provenance fields populated by intent-aware retrieval
    doc_source:   str = ""   # e.g. "sample2.pdf"
    doc_page:     int = 0    # e.g. 4
    excerpt:      str = ""   # supporting sentence from Evidence node


@dataclass
class RetrievedSubgraph:
    nodes:           List[SubgraphNode] = field(default_factory=list)
    edges:           List[SubgraphEdge] = field(default_factory=list)
    path_texts:      List[str]          = field(default_factory=list)
    evidence_chunks: List[str]          = field(default_factory=list)
    # Whether this was produced by intent-aware (targeted) retrieval
    intent_aware:    bool               = False

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def is_empty(self) -> bool:
        return not self.edges

    def merge_duplicate_edges(self):
        """Merge edges that are identical in source, relation, target, doc_source, and doc_page."""
        unique_edges = {}
        for e in self.edges:
            key = (e.source, e.relation, e.target, e.doc_source, e.doc_page)
            if key not in unique_edges:
                unique_edges[key] = e
            else:
                if e.confidence > unique_edges[key].confidence:
                    unique_edges[key] = e
        self.edges = list(unique_edges.values())
        
        # Rebuild path_texts
        self.path_texts = []
        for e in self.edges:
            pages_str = f" [p.{','.join(str(p) for p in e.pages)}]" if e.pages else ""
            if e.doc_source:
                citation = f" [{e.doc_source}, p.{e.doc_page}]" if e.doc_page else f" [{e.doc_source}]"
            else:
                citation = pages_str
            self.path_texts.append(f"{e.source} --[{e.relation}]--> {e.target}{citation}")


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_subgraph(
    linked_entities:  List[Dict],
    route:            str,
    client:           "Neo4jClient",
    max_nodes:        int = 30,
    max_edges:        int = 60,
    use_llm_cypher:   bool = False,
    question:         str = "",
    provider:         str = "auto",
    target_relations: List[str] = [],   # from QueryPlan — drives targeted Cypher
    target_labels:    List[str] = [],   # from QueryPlan — filters target node labels
) -> Tuple["RetrievedSubgraph", List[str]]:
    """
    Retrieve a subgraph around the linked entities.

    Strategy
    --------
    1. If target_relations is non-empty → _intent_aware_subgraph()
       Runs a directed, relation-filtered Cypher per seed entity and joins
       Evidence nodes to return per-edge (doc_source, doc_page, excerpt).

    2. Else → _induced_subgraph() (classic fallback)
       1–2 hop expansion, returns all edges in the induced node set.
       Used for gds_augmented, multi-hop topology queries, or when the planner
       could not identify a specific relationship intent.

    3. Optional LLM-generated Cypher (use_llm_cypher=True) tried first as an
       override; falls back to the above strategies on failure.

    Returns:
        (subgraph, generated_cyphers)
    """
    if not linked_entities:
        return RetrievedSubgraph(), []

    names              = [e["neo4j_name"] for e in linked_entities]
    generated_cyphers: List[str] = []

    # ── Optional: LLM-generated Cypher (Phase 3 override) ────────────────────
    if use_llm_cypher and question:
        try:
            from graph.schema_inspector    import get_live_schema
            from pipeline.cypher_generator import generate_cypher, execute_first_nonempty

            schema  = get_live_schema(client)
            queries = generate_cypher(
                question=question, linked_entities=linked_entities,
                schema=schema, route=route,
                provider=provider, limit=max_edges,
            )
            generated_cyphers = queries
            if queries:
                params = {"limit": max_edges}
                for i, e in enumerate(linked_entities[:3], 1):
                    params[f"entity{i}"] = e["neo4j_name"]
                rows = execute_first_nonempty(queries, client, params)
                if rows:
                    subgraph = _build_subgraph_from_rows(rows)
                    return subgraph, generated_cyphers
        except Exception as exc:
            print(f"[graph_retriever] LLM Cypher failed ({exc}), falling through")

    # ── Intent-aware targeted retrieval ──────────────────────────────────────
    if target_relations:
        print(
            f"[graph_retriever] Intent-aware mode: "
            f"relations={target_relations}, labels={target_labels}"
        )
        subgraph = _intent_aware_subgraph(
            names, client,
            target_relations=target_relations,
            target_labels=target_labels,
            max_edges=max_edges,
        )
        subgraph.intent_aware = True
        # Evidence chunks (chunk_ids) are already embedded per-edge; also expose list
        subgraph.evidence_chunks = list({
            f"{e.doc_source}:{e.doc_page}"
            for e in subgraph.edges
            if e.doc_source
        })
        return subgraph, generated_cyphers

    # ── Fallback: induced-subgraph (classic) ─────────────────────────────────
    multi_hop = route in ("hybrid_multi", "gds_augmented", "multi_hop")
    subgraph  = _induced_subgraph(names, client, multi_hop, max_nodes, max_edges)

    if subgraph.edges:
        subgraph.evidence_chunks = _fetch_evidence_chunk_ids(names, client)

    return subgraph, generated_cyphers


def retrieve_graph_context(
    linked_entities:  List[Dict],
    route:            str,
    client:           "Neo4jClient",
    max_paths:        int = 20,
    use_llm_cypher:   bool = False,
    question:         str = "",
    provider:         str = "auto",
    target_relations: List[str] = [],
    target_labels:    List[str] = [],
) -> Tuple[List[Dict], List[str]]:
    """
    Backward-compatible wrapper around retrieve_subgraph().
    Returns (path_results, generated_cyphers) matching the old signature.
    """
    subgraph, cyphers = retrieve_subgraph(
        linked_entities  = linked_entities,
        route            = route,
        client           = client,
        max_nodes        = max_paths,
        max_edges        = max_paths * 2,
        use_llm_cypher   = use_llm_cypher,
        question         = question,
        provider         = provider,
        target_relations = target_relations,
        target_labels    = target_labels,
    )
    path_results = []
    for edge in subgraph.edges:
        pages_str = f" [p.{','.join(str(p) for p in edge.pages)}]" if edge.pages else ""
        path_text = f"{edge.source} --[{edge.relation}]--> {edge.target}{pages_str}"
        path_results.append({
            "path_text":  path_text,
            "source":     edge.source,
            "relation":   edge.relation,
            "target":     edge.target,
            "confidence": edge.confidence,
            "sources":    edge.sources,
            "pages":      edge.pages,
            "stage":      edge.stage,
            "hops":       1,
            "type":       "graph",
            "doc_source": edge.doc_source,
            "doc_page":   edge.doc_page,
            "excerpt":    edge.excerpt,
        })
    return path_results, cyphers


# ── Intent-aware targeted Cypher ─────────────────────────────────────────────

def _intent_aware_subgraph(
    names:            List[str],
    client:           "Neo4jClient",
    target_relations: List[str],
    target_labels:    List[str],
    max_edges:        int,
) -> "RetrievedSubgraph":
    """
    Execute a targeted Cypher query that:
      1. Traverses ONLY the requested relation types from every seed node.
      2. Optionally filters target nodes by label.
      3. Joins Evidence nodes in the same query to fetch (doc_source, doc_page,
         excerpt) per edge — no second round-trip needed.
      4. Union-merges results from all seed names, sorted by confidence DESC.

    Falls back to an Evidence-less variant if the Evidence join returns nothing.
    """
    # Build safe relation-type string for Cypher (already validated upstream)
    rel_types = "|".join(target_relations)  # e.g. "CAUSES|HAS_SYMPTOM"

    # Core targeted query with Evidence join
    targeted_cypher = f"""
    UNWIND $names AS seed_name
    MATCH (seed {{name: seed_name}})-[r:{rel_types}]->(target)
    WHERE r.confidence > $min_conf
      AND NOT target:Chunk AND NOT target:Document AND NOT target:Evidence
      AND (size($target_labels) = 0 OR any(lbl IN labels(target) WHERE lbl IN $target_labels))
    OPTIONAL MATCH (seed)-[:HAS_EVIDENCE]->(ev:Evidence)
              -[:FROM_CHUNK]->(c:Chunk)
    WHERE ev.predicate IN $target_relations
      AND ev.object = target.name
    RETURN DISTINCT
        seed.name                            AS source,
        labels(seed)[0]                      AS source_label,
        coalesce(seed.confidence, 0.8)       AS source_conf,
        seed.pages                           AS source_pages,
        seed.umls_cui                        AS source_umls,
        seed.mesh_id                         AS source_mesh,
        type(r)                              AS relation,
        coalesce(r.confidence, 0.8)          AS rel_conf,
        r.sources                            AS sources,
        r.page_numbers                       AS pages,
        r.stage                              AS stage,
        target.name                          AS target,
        labels(target)[0]                    AS target_label,
        coalesce(target.confidence, 0.8)     AS target_conf,
        coalesce(c.source, '')               AS doc_source,
        coalesce(c.page, 0)                  AS doc_page,
        coalesce(ev.excerpt, '')             AS excerpt
    ORDER BY rel_conf DESC
    LIMIT $max_edges
    """

    rows = client.run_query(targeted_cypher, {
        "names":           names,
        "min_conf":        _CONFIDENCE_THRESHOLD,
        "target_labels":   target_labels,
        "target_relations": target_relations,
        "max_edges":       max_edges,
    })

    if rows:
        print(
            f"[graph_retriever] Targeted query returned {len(rows)} rows "
            f"for relations={target_relations}"
        )
        return _rows_to_subgraph(rows, names, include_provenance=True)

    # ── Fallback: same traversal but without Evidence join ───────────────────
    # Handles the case where Evidence nodes haven't been written yet.
    print(
        f"[graph_retriever] Targeted+Evidence query empty; "
        f"retrying without Evidence join for relations={target_relations}"
    )
    fallback_cypher = f"""
    UNWIND $names AS seed_name
    MATCH (seed {{name: seed_name}})-[r:{rel_types}]->(target)
    WHERE r.confidence > $min_conf
      AND NOT target:Chunk AND NOT target:Document AND NOT target:Evidence
      AND (size($target_labels) = 0 OR any(lbl IN labels(target) WHERE lbl IN $target_labels))
    RETURN DISTINCT
        seed.name                            AS source,
        labels(seed)[0]                      AS source_label,
        coalesce(seed.confidence, 0.8)       AS source_conf,
        seed.pages                           AS source_pages,
        seed.umls_cui                        AS source_umls,
        seed.mesh_id                         AS source_mesh,
        type(r)                              AS relation,
        coalesce(r.confidence, 0.8)          AS rel_conf,
        r.sources                            AS sources,
        r.page_numbers                       AS pages,
        r.stage                              AS stage,
        target.name                          AS target,
        labels(target)[0]                    AS target_label,
        coalesce(target.confidence, 0.8)     AS target_conf,
        ''                                   AS doc_source,
        0                                    AS doc_page,
        ''                                   AS excerpt
    ORDER BY rel_conf DESC
    LIMIT $max_edges
    """
    rows = client.run_query(fallback_cypher, {
        "names":         names,
        "min_conf":      _CONFIDENCE_THRESHOLD,
        "target_labels": target_labels,
        "max_edges":     max_edges,
    })

    if rows:
        print(
            f"[graph_retriever] Fallback (no Evidence) returned {len(rows)} rows"
        )
        return _rows_to_subgraph(rows, names, include_provenance=True)

    # Nothing found at all — return empty and let the caller fall back to vector
    print(
        f"[graph_retriever] No targeted results for "
        f"entities={names}, relations={target_relations}"
    )
    return RetrievedSubgraph()


# ── Induced-subgraph Cypher (classic fallback) ────────────────────────────────

def _induced_subgraph(
    names:      List[str],
    client:     "Neo4jClient",
    multi_hop:  bool,
    max_nodes:  int,
    max_edges:  int,
) -> "RetrievedSubgraph":
    """
    Classic induced-subgraph expansion used when target_relations is empty.
    Collects seed nodes, expands 1 or 2 hops, returns all edges in the
    induced node set.
    """
    hop_depth = 2 if multi_hop else 1

    if hop_depth == 1:
        collect_cypher = """
        MATCH (seed)
        WHERE seed.name IN $names
        OPTIONAL MATCH (seed)-[r_seed]-(hop1)
        WHERE r_seed.confidence > $min_conf
          AND NOT hop1:Chunk AND NOT hop1:Document AND NOT hop1:Evidence
        WITH collect(DISTINCT seed) + collect(DISTINCT hop1) AS node_list
        UNWIND node_list AS n
        WITH collect(DISTINCT n) AS node_set
        MATCH (a)-[r]-(b)
        WHERE a IN node_set AND b IN node_set
          AND r.confidence > $min_conf
          AND NOT a:Chunk AND NOT b:Chunk
          AND NOT a:Evidence AND NOT b:Evidence
        RETURN DISTINCT
            a.name          AS source,
            labels(a)[0]    AS source_label,
            coalesce(a.confidence, 0.8) AS source_conf,
            a.pages         AS source_pages,
            a.umls_cui      AS source_umls,
            a.mesh_id       AS source_mesh,
            type(r)         AS relation,
            coalesce(r.confidence, 0.8) AS rel_conf,
            r.sources       AS sources,
            r.page_numbers  AS pages,
            r.stage         AS stage,
            b.name          AS target,
            labels(b)[0]    AS target_label,
            coalesce(b.confidence, 0.8) AS target_conf,
            ''              AS doc_source,
            0               AS doc_page,
            ''              AS excerpt
        ORDER BY rel_conf DESC
        LIMIT $max_edges
        """
    else:
        collect_cypher = """
        MATCH (seed)
        WHERE seed.name IN $names
        OPTIONAL MATCH (seed)-[r1]-(hop1)
        WHERE r1.confidence > $min_conf
          AND NOT hop1:Chunk AND NOT hop1:Document AND NOT hop1:Evidence
        OPTIONAL MATCH (hop1)-[r2]-(hop2)
        WHERE r2.confidence > $min_conf
          AND NOT hop2:Chunk AND NOT hop2:Document AND NOT hop2:Evidence
        WITH collect(DISTINCT seed)
           + collect(DISTINCT hop1)
           + collect(DISTINCT hop2) AS node_list
        UNWIND node_list AS n
        WITH collect(DISTINCT n) AS node_set
        MATCH (a)-[r]-(b)
        WHERE a IN node_set AND b IN node_set
          AND r.confidence > $min_conf
          AND NOT a:Chunk AND NOT b:Chunk
          AND NOT a:Evidence AND NOT b:Evidence
        RETURN DISTINCT
            a.name          AS source,
            labels(a)[0]    AS source_label,
            coalesce(a.confidence, 0.8) AS source_conf,
            a.pages         AS source_pages,
            a.umls_cui      AS source_umls,
            a.mesh_id       AS source_mesh,
            type(r)         AS relation,
            coalesce(r.confidence, 0.8) AS rel_conf,
            r.sources       AS sources,
            r.page_numbers  AS pages,
            r.stage         AS stage,
            b.name          AS target,
            labels(b)[0]    AS target_label,
            coalesce(b.confidence, 0.8) AS target_conf,
            ''              AS doc_source,
            0               AS doc_page,
            ''              AS excerpt
        ORDER BY rel_conf DESC
        LIMIT $max_edges
        """

    rows = client.run_query(collect_cypher, {
        "names":     names,
        "min_conf":  _CONFIDENCE_THRESHOLD,
        "max_edges": max_edges,
    })
    return _rows_to_subgraph(rows, names, include_provenance=False)


def _fetch_evidence_chunk_ids(names: List[str], client: "Neo4jClient") -> List[str]:
    """Retrieve chunk_ids from Evidence nodes that support the seed entities."""
    try:
        rows = client.run_query("""
            MATCH (e)-[:HAS_EVIDENCE]->(ev:Evidence)-[:FROM_CHUNK]->(c:Chunk)
            WHERE e.name IN $names
            RETURN DISTINCT c.chunk_id AS chunk_id
            LIMIT 20
        """, {"names": names})
        return [r["chunk_id"] for r in rows if r.get("chunk_id")]
    except Exception:
        return []


# ── Row → Subgraph builders ───────────────────────────────────────────────────

def _rows_to_subgraph(
    rows:               List[Dict],
    seed_names:         List[str],
    include_provenance: bool = False,
) -> "RetrievedSubgraph":
    """Convert raw Cypher rows to a RetrievedSubgraph (nodes + edges + path_texts)."""
    node_map:   Dict[str, SubgraphNode] = {}
    edges:      List[SubgraphEdge]      = []
    path_texts: List[str]               = []

    for row in rows:
        src = row.get("source",  "") or ""
        tgt = row.get("target",  "") or ""
        rel = row.get("relation","") or ""
        if not (src and tgt and rel):
            continue

        # Collect unique nodes
        for name, label, conf, pages, umls, mesh in [
            (src, row.get("source_label",""), row.get("source_conf", 0.8),
             row.get("source_pages") or [], row.get("source_umls",""), row.get("source_mesh","")),
            (tgt, row.get("target_label",""), row.get("target_conf", 0.8), [], "", ""),
        ]:
            if name not in node_map:
                node_map[name] = SubgraphNode(
                    name=name, label=label or "Entity",
                    confidence=float(conf or 0.8),
                    pages=list(pages or []),
                    umls_cui=umls or "", mesh_id=mesh or "",
                )

        doc_source = row.get("doc_source", "") or "" if include_provenance else ""
        doc_page   = int(row.get("doc_page",  0) or 0) if include_provenance else 0
        excerpt    = row.get("excerpt",    "") or "" if include_provenance else ""

        edge = SubgraphEdge(
            source     = src,
            relation   = rel,
            target     = tgt,
            confidence = float(row.get("rel_conf") or 0.8),
            pages      = list(row.get("pages")   or []),
            sources    = list(row.get("sources") or []),
            stage      = row.get("stage") or "base_graph",
            doc_source = doc_source,
            doc_page   = doc_page,
            excerpt    = excerpt,
        )
        edges.append(edge)

        # Build path_text with inline citation when provenance is available
        pages_str = f" [p.{','.join(str(p) for p in edge.pages)}]" if edge.pages else ""
        if doc_source:
            citation = f" [{doc_source}, p.{doc_page}]" if doc_page else f" [{doc_source}]"
        else:
            citation = pages_str
        path_texts.append(f"{src} --[{rel}]--> {tgt}{citation}")

    return RetrievedSubgraph(
        nodes      = list(node_map.values()),
        edges      = edges,
        path_texts = path_texts,
    )


def _build_subgraph_from_rows(rows: List[Dict]) -> "RetrievedSubgraph":
    """Build a subgraph from LLM-generated Cypher result rows."""
    return _rows_to_subgraph(rows, [], include_provenance=False)


# ── Context formatters ────────────────────────────────────────────────────────

def format_subgraph_for_context(subgraph: "RetrievedSubgraph") -> str:
    """
    Render the subgraph as a structured context block for the LLM.

    Intent-aware subgraphs include per-edge provenance citations so the LLM
    can produce grounded answers like:
        "Metformin causes Nausea [sample2.pdf, p.4]"
    """
    if subgraph.is_empty():
        return ""

    mode = "INTENT-AWARE" if subgraph.intent_aware else "INDUCED-SUBGRAPH"
    lines = [
        f"[KNOWLEDGE SUBGRAPH ({mode}) — "
        f"{subgraph.node_count} nodes, {subgraph.edge_count} edges]"
    ]

    # Node summary
    node_parts = []
    for n in subgraph.nodes[:20]:
        umls_str = f" [UMLS:{n.umls_cui}]" if n.umls_cui else ""
        node_parts.append(f"{n.name} ({n.label}{umls_str})")
    lines.append("Nodes: " + ", ".join(node_parts))

    # Edges with inline citations
    lines.append("Edges:")
    for e in subgraph.edges[:40]:
        conf_str = f"conf={e.confidence:.2f}"

        # Build citation: prefer doc_source/doc_page from Evidence join;
        # fall back to relation-level page_numbers
        if e.doc_source:
            cite = f"{e.doc_source}, p.{e.doc_page}" if e.doc_page else e.doc_source
        elif e.pages:
            cite = f"p.{','.join(str(p) for p in e.pages[:3])}"
        else:
            cite = ""

        detail = ", ".join(filter(None, [conf_str, cite]))
        ev_str = f'  excerpt: "{e.excerpt[:120]}"' if e.excerpt else ""
        lines.append(f"  {e.source} --[{e.relation}]--> {e.target}  ({detail}){ev_str}")

    if subgraph.evidence_chunks:
        lines.append(
            f"Evidence anchors: {len(subgraph.evidence_chunks)} source(s) — "
            + ", ".join(sorted(subgraph.evidence_chunks)[:5])
        )

    return "\n".join(lines)


def format_for_context(graph_results: List[Dict]) -> str:
    """Legacy path-list formatter (backward-compatible)."""
    if not graph_results:
        return ""
    lines = ["[GRAPH KNOWLEDGE]"]
    for r in graph_results:
        conf_str  = f" (conf={r.get('confidence',0):.2f})" if r.get("confidence") else ""
        pages_str = (
            f" [p.{','.join(str(p) for p in r['pages'])}]" if r.get("pages") else ""
        )
        lines.append(f"  • {r['path_text']}{conf_str}{pages_str}")
    return "\n".join(lines)
