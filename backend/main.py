"""
backend/main.py
Phase 1-6 — FastAPI application with full enhanced pipeline.

Enhancements integrated:
  Phase 2: identifier linking flag in /ingest
  Phase 3: use_llm_cypher flag, generated_cypher in response
  Phase 4: use_gds flag, GDS context in pipeline
  Phase 5: Qdrant-first retrieval ordering + chunk-entity enrichment
  Phase 6: document registry (SHA-256 dedup), /documents endpoint

Endpoints:
  POST /query          → Ask a medical question (full hybrid pipeline)
  POST /ingest         → Upload + ingest a PDF
  GET  /documents      → List all ingested documents
  GET  /graph/explore  → Browse Neo4j graph nodes + edges
  GET  /graph/subgraph → Induced subgraph for a list of entity names (Phase 7 frontend)
  GET  /graph/node/{name} → Full provenance for a single node
  GET  /graph/stats    → Node + edge counts
  GET  /health         → Health check
"""

from __future__ import annotations
import os
import sys
import time
from typing import List, Optional, Dict

import neo4j.exceptions
import openai
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.neo4j_client         import Neo4jClient, NODE_LABELS
from llm                        import RateLimitExhausted
from pipeline.query_planner     import plan_query
from pipeline.query_rewriter    import rewrite_query, no_rewrite
from pipeline.entity_linker     import link_entities, invalidate_cache as invalidate_entity_cache
from pipeline.graph_retriever   import retrieve_subgraph, format_subgraph_for_context
from pipeline.vector_retriever  import retrieve_vector_context
from pipeline.reranker          import fuse_and_rerank
from pipeline.generator         import generate_answer
from pipeline.completeness      import MAX_REPAIR_PARTS, check_coverage, should_check
from vector.qdrant_client       import VectorStoreUnavailable

app = FastAPI(
    title="Medical Graph RAG API",
    description=(
        "Production-grade Hybrid Graph + Vector RAG for multi-hop medical question answering. "
        "Combines Neo4j knowledge graph with Qdrant semantic search, GDS algorithms, "
        "dynamic Cypher generation, and complete provenance tracking."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built frontend (Phase 7) — skipped gracefully if not built yet
_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/app", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")

@app.exception_handler(RateLimitExhausted)
def rate_limit_exhausted_handler(_request: Request, exc: RateLimitExhausted):
    """Turn an exhausted LLM quota into a clear 429 instead of a generic 500."""
    retry_hint = (f" Try again in about {exc.wait_seconds / 60:.0f} minutes."
                  if exc.wait_seconds else " Try again later.")
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"The {exc.provider} LLM rate limit is used up.{retry_hint} "
                      "For ingestion, chunks finished so far are cached and will be "
                      "skipped on the next upload.",
            "provider": exc.provider,
            "retry_after_seconds": exc.wait_seconds,
        },
    )


@app.exception_handler(openai.APIStatusError)
def llm_api_error_handler(_request: Request, exc: openai.APIStatusError):
    """Invalid API key, unknown model, etc. — tell the user what the provider said."""
    hints = {401: " Check the API key in .env.",
             404: " Check the model name (<PROVIDER>_MODEL) in .env."}
    return JSONResponse(
        status_code=502,
        content={"detail": f"LLM provider error ({exc.status_code}): {exc.message}."
                           f"{hints.get(exc.status_code, '')}"},
    )


def _store_unavailable(service: str, hint: str, exc: Exception) -> JSONResponse:
    print(f"[main] {service} unavailable: {exc}")
    return JSONResponse(
        status_code=503,
        content={"detail": f"{service} is not reachable. {hint}", "service": service},
    )


@app.exception_handler(VectorStoreUnavailable)
def qdrant_unavailable_handler(_request: Request, exc: VectorStoreUnavailable):
    """Qdrant down → 503 with a hint, instead of a bare 500."""
    return _store_unavailable(
        "Qdrant", "Start it (docker run -p 6333:6333 qdrant/qdrant) or check QDRANT_URL in .env.", exc)


@app.exception_handler(neo4j.exceptions.ServiceUnavailable)
def neo4j_unavailable_handler(_request: Request, exc: neo4j.exceptions.ServiceUnavailable):
    return _store_unavailable("Neo4j", "Start it or check NEO4J_URI in .env.", exc)


@app.exception_handler(neo4j.exceptions.AuthError)
def neo4j_auth_handler(_request: Request, exc: neo4j.exceptions.AuthError):
    return _store_unavailable("Neo4j", "Authentication failed: check NEO4J_USER / NEO4J_PASSWORD in .env.", exc)


# Shared Neo4j client (connection pool)
_neo4j: Optional[Neo4jClient] = None


def get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
    return _neo4j


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:        str
    provider:        str   = "auto"
    top_k:           int   = 5
    top_n:           int   = 8
    use_query_rewrite: bool = True    # rewrite + split the question before retrieval
    use_completeness_check: bool = True  # check every part is answered; repair once if not
    use_llm_cypher:  bool  = False    # Phase 3: LLM-generated Cypher
    # GDS is now AUTO-ACTIVATED from route — no toggle needed;
    # set force_no_gds=True to disable for latency-sensitive requests
    force_no_gds:    bool  = False
    min_confidence:  float = 0.5     # filter graph paths below this confidence


class QueryResponse(BaseModel):
    question:            str
    rewritten_query:     str              # what retrieval actually searched for
    sub_queries:         List[str]        # extra searches for multi-part questions
    answer:              str
    route:               str
    route_reason:        str
    retrieval_strategy:  str
    planner_confidence:  float
    target_relations:    List[str]        # intent relations extracted by planner
    target_labels:       List[str]        # intent node labels extracted by planner
    intent_aware:        bool             # True when targeted Cypher was used
    linked_entities:     List[str]
    graph_paths:         List[str]
    gds_paths:           List[str]
    vector_sources:      List[str]
    generated_cypher:    List[str]
    provenance:          List[Dict]
    subgraph:            Optional[Dict]   # structured node+edge subgraph
    retrieved_contexts:  List[str]        # exact context pieces given to the answer LLM
    completeness:        Optional[Dict] = None  # parts found, parts missing, whether repaired
    latency_ms:          float


class GraphExploreResponse(BaseModel):
    nodes: List[dict]
    edges: List[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        neo4j  = get_neo4j()
        status = neo4j.verify_connection()
        return {"status": "ok", "neo4j": status, "version": "2.0.0"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0    = time.time()
    neo4j = get_neo4j()

    # ── Step 0: Rewrite the question for retrieval ────────────────────────────
    # Retrieval and planning use the rewritten question; the final answer
    # still responds to the user's original wording.
    rewrite = (rewrite_query(req.question, provider=req.provider)
               if req.use_query_rewrite else no_rewrite(req.question))
    search_question = rewrite.rewritten

    # ── Step 1: Qdrant-first semantic retrieval (one search per sub-query) ────
    vector_results = _merge_round_robin([
        retrieve_vector_context(q, top_k=req.top_k) for q in rewrite.search_queries
    ])

    # ── Map internal IDs to filenames ────────────────────────────────────────
    from graph.document_registry import get_source_map
    try:
        source_map = get_source_map(neo4j)
    except Exception as e:
        print(f"[main] Failed to get source map: {e}")
        source_map = {}

    for r in vector_results:
        orig = r.get("source", "")
        if orig in source_map:
            r["source"] = source_map[orig]

    # ── Step 2: Spot entities in retrieved chunks (Phase 5) ──────────────────
    chunk_entity_names: List[str] = []
    if vector_results:
        try:
            from pipeline.entity_extractor_lite import spot_entities_in_chunks
            chunk_entity_names = spot_entities_in_chunks(
                vector_results, provider=req.provider
            )
        except RateLimitExhausted:
            raise
        except Exception as e:
            print(f"[main] entity_extractor_lite failed: {e}")

    # ── Step 3: Adaptive query planning ──────────────────────────────────────
    chunk_texts = [r["text"] for r in vector_results]
    plan = plan_query(
        search_question,
        provider=req.provider,
        context_chunks=chunk_texts,
    )
    print(f"[main] Route={plan.route} conf={plan.confidence:.2f} — {plan.reason}")

    # ── Step 4: Entity linking ────────────────────────────────────────────────
    all_entity_names = list(set(plan.entities_mentioned) | set(chunk_entity_names))

    linked:            List[Dict] = []
    graph_results:     List[Dict] = []
    gds_results:       List[Dict] = []
    generated_cyphers: List[str]  = []

    # ── Step 5: Subgraph retrieval (skipped for vector_only) ─────────────────
    retrieved_subgraph = None
    if plan.route != "vector_only" and all_entity_names:
        linked = link_entities(all_entity_names, neo4j)

        if linked:
            retrieved_subgraph, generated_cyphers = retrieve_subgraph(
                linked_entities  = linked,
                route            = plan.route,
                client           = neo4j,
                max_nodes        = 30,
                max_edges        = 60,
                use_llm_cypher   = req.use_llm_cypher,
                question         = search_question,
                provider         = req.provider,
                target_relations = plan.target_relations,   # intent flows into Neo4j
                target_labels    = plan.target_labels,      # intent flows into Neo4j
            )
            
            # Map graph IDs and merge duplicates
            if retrieved_subgraph:
                for e in retrieved_subgraph.edges:
                    if e.doc_source in source_map:
                        e.doc_source = source_map[e.doc_source]
                    if e.sources:
                        e.sources = list(set(source_map.get(s, s) for s in e.sources))
                
                if retrieved_subgraph.evidence_chunks:
                    new_chunks = set()
                    for chunk in retrieved_subgraph.evidence_chunks:
                        if ":" in chunk:
                            src, page = chunk.split(":", 1)
                            new_chunks.add(f"{source_map.get(src, src)}:{page}")
                        else:
                            new_chunks.add(source_map.get(chunk, chunk))
                    retrieved_subgraph.evidence_chunks = list(new_chunks)
                
                retrieved_subgraph.merge_duplicate_edges()

            # Convert to path_text dicts for reranker — carry provenance fields
            graph_results = [
                {
                    "path_text":  pt,
                    "type":       "graph",
                    "confidence": e.confidence,
                    "pages":      e.pages,
                    "sources":    e.sources,
                    "stage":      e.stage,
                    "text":       pt,
                    "doc_source": e.doc_source,   # e.g. "sample2.pdf"
                    "doc_page":   e.doc_page,     # e.g. 4
                    "excerpt":    e.excerpt,      # supporting sentence
                }
                for e, pt in zip(retrieved_subgraph.edges, retrieved_subgraph.path_texts)
                if e.confidence >= req.min_confidence
            ]

    # ── Step 6: Auto-GDS (driven by route, not user flag) ────────────────────
    #   gds_augmented → full algorithms
    #   hybrid_multi with ≥2 entities → shortest path
    #   hybrid_single → node similarity
    #   vector_only → skip
    if not req.force_no_gds and linked and plan.route != "vector_only":
        try:
            from graph.gds_client       import GDSClient
            from pipeline.gds_retriever import retrieve_gds_context
            gds_client  = GDSClient(neo4j)
            gds_results = retrieve_gds_context(
                linked_entities = linked,
                route           = plan.route,
                neo4j_client    = neo4j,
                gds_client      = gds_client,
                mode            = "auto",   # auto-selects based on route
            )
        except Exception as e:
            print(f"[main] Auto-GDS failed (non-fatal): {e}")

    # ── Step 7: Rerank + fuse all three sources ───────────────────────────────
    ranked_items, fused_context = fuse_and_rerank(
        query          = search_question,
        graph_results  = graph_results,
        vector_results = vector_results,
        gds_results    = gds_results,
        top_n          = req.top_n,
    )

    # ── Step 8: Generate answer ───────────────────────────────────────────────
    # Use structured subgraph context when available, else fall back to path list
    retrieved_contexts = [f"{item['meta']}\n{item['text']}" for item in ranked_items]
    if retrieved_subgraph and not retrieved_subgraph.is_empty():
        subgraph_context = format_subgraph_for_context(retrieved_subgraph)
        # Prepend the structured subgraph block to the fused context
        fused_context = subgraph_context + "\n\n" + fused_context if fused_context else subgraph_context
        retrieved_contexts.insert(0, subgraph_context)
    answer  = generate_answer(req.question, fused_context, provider=req.provider)

    # ── Step 9: Completeness check + one targeted repair ─────────────────────
    completeness = None
    if req.use_completeness_check and should_check(plan.route, rewrite.sub_queries):
        answer, fused_context, completeness = _repair_incomplete_answer(
            req, rewrite.sub_queries, answer, fused_context,
            ranked_items, retrieved_contexts, source_map,
        )
    latency = (time.time() - t0) * 1000

    # Build provenance list
    provenance = []
    seen_doc_prov = {}
    for item in ranked_items:
        if item["type"] == "document":
            prov_key = (item.get("source", ""), item.get("page", 0))
            if prov_key not in seen_doc_prov:
                seen_doc_prov[prov_key] = {
                    "type":   "document",
                    "source": item.get("source", ""),
                    "page":   item.get("page",   0),
                    "text":   item["text"][:200],
                    "score":  round(item.get("final_score", 0), 3),
                }
                provenance.append(seen_doc_prov[prov_key])
            else:
                seen_doc_prov[prov_key]["score"] = max(
                    seen_doc_prov[prov_key]["score"],
                    round(item.get("final_score", 0), 3)
                )
        elif item["type"] in ("graph", "graph_gds"):
            prov_entry = {
                "type":       item["type"],
                "path":       item["text"],
                "pages":      item.get("pages", []),
                "confidence": round(item.get("confidence", 0), 3),
                "score":      round(item.get("final_score", 0), 3),
            }
            # Attach precise file+page citation when available (intent-aware)
            if item.get("doc_source"):
                prov_entry["doc_source"] = item["doc_source"]
                prov_entry["doc_page"]   = item.get("doc_page", 0)
            if item.get("excerpt"):
                prov_entry["excerpt"] = item["excerpt"][:200]
            provenance.append(prov_entry)

    return QueryResponse(
        question            = req.question,
        rewritten_query     = rewrite.rewritten,
        sub_queries         = rewrite.sub_queries,
        answer              = answer,
        route               = plan.route,
        route_reason        = plan.reason,
        retrieval_strategy  = plan.retrieval_strategy,
        planner_confidence  = round(plan.confidence, 3),
        target_relations    = plan.target_relations,
        target_labels       = plan.target_labels,
        intent_aware        = retrieved_subgraph.intent_aware if retrieved_subgraph else False,
        linked_entities     = [e["neo4j_name"] for e in linked],
        graph_paths         = [r["path_text"] for r in graph_results],
        gds_paths           = [r["path_text"] for r in gds_results],
        vector_sources      = list(dict.fromkeys(f"{r['source']} p.{r['page']}" for r in vector_results)),
        generated_cypher    = generated_cyphers,
        provenance          = provenance,
        subgraph            = {
            "nodes": [
                {"name": n.name, "label": n.label, "confidence": n.confidence,
                 "pages": n.pages, "umls_cui": n.umls_cui}
                for n in retrieved_subgraph.nodes
            ] if retrieved_subgraph else [],
            "edges": [
                {
                    "source":     e.source,
                    "relation":   e.relation,
                    "target":     e.target,
                    "confidence": e.confidence,
                    "pages":      e.pages,
                    "doc_source": e.doc_source,
                    "doc_page":   e.doc_page,
                }
                for e in retrieved_subgraph.edges
            ] if retrieved_subgraph else [],
            "evidence_chunks": retrieved_subgraph.evidence_chunks if retrieved_subgraph else [],
            "intent_aware":    retrieved_subgraph.intent_aware if retrieved_subgraph else False,
        },
        retrieved_contexts  = retrieved_contexts,
        completeness        = completeness,
        latency_ms          = round(latency, 1),
    )


def _repair_incomplete_answer(
    req:                QueryRequest,
    sub_queries:        List[str],
    answer:             str,
    fused_context:      str,
    ranked_items:       List[Dict],
    retrieved_contexts: List[str],
    source_map:         Dict[str, str],
):
    """
    Check the draft answer covers every part of the question; if a part is
    missing, search for that part specifically and regenerate once.

    Mutates ranked_items / retrieved_contexts in place so provenance and the
    reported contexts include whatever was added. Returns
    (answer, fused_context, completeness_info).
    """
    report = check_coverage(req.question, answer, sub_queries, provider=req.provider)
    info = {"parts": report.part_names, "missing": report.missing,
            "repaired": False, "added_sources": []}
    if not report.missing:
        return answer, fused_context, info

    seen  = {item["text"][:100] for item in ranked_items}
    extra: List[Dict] = []
    for part in report.missing[:MAX_REPAIR_PARTS]:
        hits = retrieve_vector_context(part, top_k=req.top_k)
        for hit in hits:
            hit["source"] = source_map.get(hit.get("source", ""), hit.get("source", ""))
        hits = [h for h in hits if h["text"][:100] not in seen]
        if not hits:
            continue
        best, _ = fuse_and_rerank(query=part, graph_results=[], vector_results=hits, top_n=2)
        for item in best:
            seen.add(item["text"][:100])
            extra.append(item)

    if extra:
        added = [f"{item['meta']}\n{item['text']}" for item in extra]
        fused_context = "\n\n".join([fused_context, *added]) if fused_context else "\n\n".join(added)
        retrieved_contexts.extend(added)
        ranked_items.extend(extra)

    # Regenerate even when nothing new was found: most misses are parts whose
    # facts were already in the context but were dropped from the answer.
    answer = generate_answer(req.question, fused_context, provider=req.provider,
                             required_parts=report.part_names)
    info["repaired"] = True
    info["added_sources"] = list(dict.fromkeys(f"{i['source']} p.{i['page']}" for i in extra))
    return answer, fused_context, info


# Plain `def` (not `async def`): ingestion blocks for minutes, so FastAPI must
# run it in a worker thread or every other request would freeze meanwhile.
def _merge_round_robin(result_lists: List[List[Dict]]) -> List[Dict]:
    """
    Merge vector results from several sub-queries, taking the best hit of each
    sub-query first, then the second-best of each, and so on (duplicates dropped).

    Interleaving matters because later steps only look at the first few chunks:
    sorting by raw score could let one sub-query fill all of those slots.
    """
    merged: List[Dict] = []
    seen_ids = set()
    for rank in range(max((len(results) for results in result_lists), default=0)):
        for results in result_lists:
            if rank < len(results) and results[rank]["chunk_id"] not in seen_ids:
                seen_ids.add(results[rank]["chunk_id"])
                merged.append(results[rank])
    return merged


@app.post("/ingest")
def ingest(
    file:         UploadFile = File(...),
    provider:     str        = Query("auto", description="LLM provider: auto, gemini, groq, openai, ollama"),
    chunk_tokens: int        = Query(512, ge=128, le=4096,
                                     description="Chunk size in tokens. 1024 halves the LLM calls for long PDFs"),
    link_ids:     bool       = Query(False,  description="Phase 2: link UMLS/MeSH/RxNorm IDs"),
    valid_from:   str        = Query("",     description="Phase 2: ISO date validity start"),
    valid_until:  str        = Query("",     description="Phase 2: ISO date validity end"),
):
    """
    Upload a PDF, run full Stage 1 pipeline, store in Neo4j + Qdrant.

    If the LLM quota runs out midway, responds 429 and nothing is written;
    finished chunks stay cached, so uploading the same PDF again resumes.
    """
    import tempfile, hashlib

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    t0           = time.time()
    file_bytes   = file.file.read()
    sha256       = hashlib.sha256(file_bytes).hexdigest()
    neo4j        = get_neo4j()

    # Phase 6: check document registry for duplicate
    from graph.document_registry import is_already_ingested, register_document
    if is_already_ingested(sha256, neo4j):
        return {
            "status":   "already_ingested",
            "file":     file.filename,
            "sha256":   sha256[:16] + "…",
            "message":  "This exact file has already been processed. Skipping.",
        }

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        from ingestion.pdf_parser       import parse_pdf
        from ingestion.chunker          import chunk_pages
        from ingestion.entity_extractor import extract_from_chunks
        from ingestion.deduplicator     import deduplicate
        from ingestion.embedder         import embed_chunks
        from graph.graph_loader         import load_to_graph, load_chunks_to_qdrant

        pages   = parse_pdf(tmp_path)
        chunks  = chunk_pages(pages, chunk_tokens=chunk_tokens)

        # LLM extraction first: it is the step that can hit rate limits, and if
        # it does we want to stop before anything is written to Neo4j/Qdrant.
        results             = extract_from_chunks(chunks, provider=provider)
        entities, relations = deduplicate(results)
        chunks              = embed_chunks(chunks)

        # Phase 2: optional identifier linking
        if link_ids:
            try:
                from ingestion.identifier_linker import link_identifiers
                entities = link_identifiers(entities, verbose=False)
            except Exception as e:
                print(f"[ingest] identifier_linker failed (non-fatal): {e}")

        # Build full Document → Chunk → Entity provenance graph
        counts = load_to_graph(
            chunks             = chunks,
            entities           = entities,
            relations          = relations,
            extraction_results = results,
            client             = neo4j,
            filename           = file.filename,
            sha256             = sha256,        # canonical document identity
        )
        load_chunks_to_qdrant(chunks)

        # Phase 2: stamp temporal validity on all relations if provided
        if valid_from or valid_until:
            for rel in relations:
                try:
                    neo4j.set_relation_temporal(
                        subj        = rel["subject"],
                        pred        = rel["predicate"],
                        obj         = rel["object"],
                        valid_from  = valid_from  or None,
                        valid_until = valid_until or None,
                    )
                except Exception:
                    pass

        # Phase 6: register document
        register_document(
            sha256         = sha256,
            filename       = file.filename,
            client         = neo4j,
            page_count     = len(pages),
            entity_count   = counts["entities"],
            relation_count = counts["relations"],
        )

        invalidate_entity_cache()   # let queries link the newly added entities

        latency = (time.time() - t0) * 1000
        return {
            "status":    "ok",
            "file":      file.filename,
            "sha256":    sha256[:16] + "…",
            "pages":     len(pages),
            "chunks":    len(chunks),
            "entities":  counts["entities"],
            "relations": counts["relations"],
            "latency_ms": round(latency, 1),
        }
    finally:
        os.unlink(tmp_path)


# ── Phase 6: Document registry ────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    """Return all ingested documents with metadata."""
    from graph.document_registry import list_documents as _list
    return {"documents": _list(get_neo4j())}


@app.post("/documents/purge-legacy")
def purge_legacy_documents():
    """
    One-shot cleanup: detach-delete all Document nodes that have no sha256.
    These were created by old code that keyed Document nodes by filename only.
    Safe to call at any time — sha256-keyed nodes are never affected.
    """
    from graph.document_registry import purge_legacy_documents as _purge
    result = _purge(get_neo4j())
    return {"status": "ok", **result}


@app.delete("/documents/{sha256}")
def delete_document(sha256: str):
    """
    Delete a document's Neo4j graph data (Document node + exclusive Chunks)
    and all matching Qdrant embeddings (filtered by source filename).

    Entity/Relationship nodes shared with other documents are preserved.
    """
    from graph.document_registry import delete_document as _delete_neo4j
    from vector.qdrant_client import QdrantStore

    neo4j = get_neo4j()

    # Qdrant points are tagged with the chunk "source" (the temp-file name used
    # at upload time), not the original filename — collect those before the
    # chunks are deleted from Neo4j.
    chunk_sources = [
        row["source"] for row in neo4j.run_query(
            "MATCH (:Document {sha256: $sha256})-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN DISTINCT c.source AS source",
            {"sha256": sha256},
        ) if row.get("source")
    ]

    # Step 1: purge Neo4j data
    neo4j_result = _delete_neo4j(sha256, neo4j)
    if neo4j_result.get("error"):
        raise HTTPException(status_code=404, detail=neo4j_result["error"])

    filename = neo4j_result["filename"]
    invalidate_entity_cache()

    # Step 2: purge Qdrant embeddings for this document
    qdrant_deleted = 0
    try:
        store = QdrantStore()
        for source in chunk_sources or [filename]:
            qdrant_deleted += store.delete_by_source(source)
    except Exception as e:
        print(f"[delete_document] Qdrant purge failed (non-fatal): {e}")

    return {
        "status":          "deleted",
        "filename":        filename,
        "sha256":          sha256[:16] + "…",
        "neo4j": {
            "doc_deleted":    neo4j_result["doc_deleted"],
            "chunks_deleted": neo4j_result["chunks_deleted"],
        },
        "qdrant": {
            "points_deleted": qdrant_deleted,
        },
    }


# ── Graph exploration ─────────────────────────────────────────────────────────

_EXPLORE_NODE_FIELDS = (
    "RETURN n.name AS name, labels(n)[0] AS label, "
    "n.confidence AS confidence, n.stage AS stage, "
    "n.pages AS pages, n.sources AS sources, "
    "n.umls_cui AS umls_cui, n.mesh_id AS mesh_id, n.rxnorm_id AS rxnorm_id"
)


@app.get("/graph/explore", response_model=GraphExploreResponse)
def graph_explore(
    label:        Optional[str] = Query(None, description="Filter by node label"),
    limit:        int           = Query(50,   description="Max nodes to return"),
    min_conf:     float         = Query(0.0,  description="Min node confidence filter"),
):
    """Browse graph nodes and their direct relationships."""
    # Unknown labels get a clear 400 instead of an empty graph.
    allowed_labels = NODE_LABELS + ["Document", "Chunk", "Evidence"]
    if label and label not in allowed_labels:
        raise HTTPException(status_code=400,
                            detail=f"Unknown label '{label}'. Valid: {', '.join(allowed_labels)}")
    neo4j = get_neo4j()

    # Only named entity nodes are drawn. Chunk/Document/Evidence nodes have no
    # `name`, so without this filter they used up the node budget for nothing.
    labels      = [label] if label else NODE_LABELS
    conf_filter = "AND n.confidence >= $min_conf" if min_conf > 0 else ""
    nodes = neo4j.run_query(
        "MATCH (n) WHERE n.name IS NOT NULL AND any(l IN labels(n) WHERE l IN $labels) "
        f"{conf_filter} {_EXPLORE_NODE_FIELDS} LIMIT $limit",
        {"labels": labels, "limit": limit, "min_conf": min_conf},
    )
    names = [n["name"] for n in nodes]

    # Every edge must end on a node the client receives, or Cytoscape refuses
    # to draw the graph at all. Unfiltered: edges inside the node set only.
    # Filtered by label: also edges to neighbours, whose nodes are added below.
    edges = neo4j.run_query(
        """
        MATCH (a)-[r]->(b)
        WHERE a.name IS NOT NULL AND b.name IS NOT NULL
          AND any(l IN labels(a) WHERE l IN $entity_labels)
          AND any(l IN labels(b) WHERE l IN $entity_labels)
          AND r.confidence >= $min_conf
          AND ((a.name IN $names AND b.name IN $names)
               OR ($with_neighbours AND (a.name IN $names OR b.name IN $names)))
        RETURN a.name AS source, type(r) AS relation, b.name AS target,
               r.confidence AS confidence, r.stage AS stage,
               r.page_numbers AS pages, r.sources AS sources
        LIMIT $edge_limit
        """,
        {"names": names, "entity_labels": NODE_LABELS, "min_conf": min_conf,
         "with_neighbours": bool(label), "edge_limit": limit * 3},
    )

    known      = set(names)
    neighbours = list(dict.fromkeys(
        end for e in edges for end in (e["source"], e["target"]) if end not in known))
    if label and neighbours:
        nodes += neo4j.run_query(
            f"MATCH (n) WHERE n.name IN $names {_EXPLORE_NODE_FIELDS}", {"names": neighbours})

    # The UI uses the name as the node id, so one node per name.
    unique = {}
    for n in nodes:
        unique.setdefault(n["name"], n)
    nodes = list(unique.values())
    known = set(unique)
    edges = [e for e in edges if e["source"] in known and e["target"] in known]

    return GraphExploreResponse(nodes=nodes, edges=edges)


@app.get("/graph/subgraph")
def graph_subgraph(
    entities: str = Query(..., description="Comma-separated entity names"),
    depth:    int = Query(1,   description="Traversal depth (1 or 2)"),
):
    """
    Return the induced subgraph around a set of named entities.
    Used by the Phase 7 frontend to highlight query-relevant nodes.
    """
    neo4j = get_neo4j()
    names = [n.strip() for n in entities.split(",") if n.strip()]
    if not names:
        return {"nodes": [], "edges": []}

    depth = min(depth, 2)  # cap at 2 hops for performance
    cypher = f"""
    UNWIND $names AS name
    MATCH (n {{name: name}})-[r*1..{depth}]-(m)
    WITH collect(DISTINCT n) + collect(DISTINCT m) AS all_nodes,
         collect(DISTINCT r) AS all_rels
    UNWIND all_nodes AS node
    WITH collect(DISTINCT {{name: node.name, label: labels(node)[0],
                             confidence: node.confidence, stage: node.stage}}) AS nodes,
         all_rels
    UNWIND all_rels AS rel_list
    UNWIND rel_list AS rel
    WITH nodes,
         collect(DISTINCT {{
           source: startNode(rel).name,
           target: endNode(rel).name,
           relation: type(rel),
           confidence: rel.confidence
         }}) AS edges
    RETURN nodes, edges
    """
    try:
        rows = neo4j.run_query(cypher, {"names": names})
        if rows:
            return {"nodes": rows[0].get("nodes", []),
                    "edges": rows[0].get("edges", [])}
    except Exception as e:
        print(f"[graph_subgraph] Error: {e}")

    return {"nodes": [], "edges": []}


@app.get("/graph/node/{name}")
def graph_node_provenance(name: str):
    """Return full provenance for a single named entity (Phase 7 click-to-inspect)."""
    neo4j = get_neo4j()
    return neo4j.get_provenance(name)


@app.get("/graph/stats")
def graph_stats():
    """Return node and edge counts per type."""
    neo4j = get_neo4j()
    return {
        "nodes": neo4j.get_node_counts(),
        "edges": neo4j.get_edge_counts(),
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
