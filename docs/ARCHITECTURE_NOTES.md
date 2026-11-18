# Architecture Notes — Medical Graph RAG

Study notes on how this codebase actually works, written from the code (not the
README). File references are `path:line` or `path` + function name.

Original project by **DongaShanmukhaSivaVenkataSai**
(github.com/DongaShanmukhaSivaVenkataSai/Medical-Graph-RAG-for-Clinical-Reasoning-).

The notes describe the code **after** the fork's changes. Where the original
behaved differently, it says so ("Before this fork: …"). §11 lists every change.

---

## 1. The system in one picture

```
                         ┌──────────────── INGESTION (POST /ingest, run_stage1.py) ───────────────┐
 PDF ─► parse_pdf ─► chunk_pages ─► extract_from_chunks (LLM, cached) ─► deduplicate ─► Neo4j
                         │                                                              ▲
                         └─► embed_chunks (BGE) ─────────────────────────────► Qdrant   │
                                                                                         │
                         ┌──────────────── QUERY (POST /query) ──────────────────────────┤
 question ─► rewrite ─► vector search per sub-query ─► spot entities in chunks ─► plan route
            ─► link entities ─► graph retrieval (+ GDS) ─► cross-encoder rerank ─► answer LLM
```

Four processes: **FastAPI** (`backend/main.py`), **Neo4j 5** (knowledge graph +
provenance), **Qdrant** (chunk vectors), **React/Vite** (`frontend/`). Every LLM call
in the project goes through one module, `llm/client.py`.

---

## 2. Ingestion pipeline

Entry points: `POST /ingest` (`backend/main.py:424`, `ingest()`) and the CLI
`run_stage1.py`. Same steps, in this order:

| # | Step | Code | What happens |
|---|---|---|---|
| 0 | Dedup check | `graph/document_registry.py` `is_already_ingested` | SHA-256 of the file bytes; if a `Document {sha256}` exists, return `already_ingested`. |
| 1 | Parse | `ingestion/pdf_parser.py` `parse_pdf` | PyMuPDF, one dict per page `{page, text, source, total_pages}`. `source` = file stem. For API uploads the stem is a **temp-file name** (see §7). |
| 2 | Chunk | `ingestion/chunker.py` `chunk_pages` | Character sliding window, ~4 chars/token, default 512 tokens with 100-token overlap, **never crosses a page boundary** (so every chunk has exactly one page number). `chunk_id = md5(source_p{page}_c{idx})[:12]`. |
| 3 | Extract | `ingestion/entity_extractor.py` `extract_from_chunks` → `extract_from_chunk` | One LLM call per chunk, JSON mode. 15 entity types, 16 relation types (prompt at `:158`). |
| 3a | Validate | `_parse_and_validate` (`:278`) | Pydantic drops unknown entity types; unknown predicates are coerced to `RELATED_TO`. Then 4 filters: entity confidence ≥ 0.40, **entity grounding** (`_is_grounded`, `:121`: name/alias substring or all ≥4-char words present in the chunk text), relation confidence ≥ 0.50, **ontology check** (`ingestion/ontology_validator.py` `validate_triple`: domain/range per predicate, e.g. `HAS_SYMPTOM` must be Disease→Symptom). |
| 3b | Cache | `_load_cached` / `_save_cached` | Result per chunk saved to `.cache/extractions/<key>.json`, where the key hashes the chunk text **and** a fingerprint of the extraction setup (see §6). |
| 4 | Deduplicate | `ingestion/deduplicator.py` `deduplicate` | `normalize_name` (NFKC, abbreviation map `t2dm→Type 2 Diabetes`, title-case, keep short acronyms), alias merge, then rapidfuzz `ratio ≥ 90` merge. Relation endpoints are remapped to canonical names; pages/chunk_ids are unioned. |
| 5 | Embed | `ingestion/embedder.py` `embed_chunks` | `BAAI/bge-base-en-v1.5`, L2-normalised. Queries get the BGE instruction prefix (`embed_query`). |
| 6 | Optional ID linking | `ingestion/identifier_linker.py` | UMLS/MeSH/RxNorm IDs if `link_ids=true`. |
| 7 | Write graph | `graph/graph_loader.py` `load_to_graph` | indexes → entity nodes → relation edges → Chunk nodes → `Document-[:HAS_CHUNK]->Chunk` → `Chunk-[:MENTIONS]->Entity` → Evidence nodes (`graph/evidence_writer.py`). |
| 8 | Write vectors | `load_chunks_to_qdrant` → `vector/qdrant_client.py` | Point id = `uuid5(chunk_id)` (idempotent), cosine distance, payload `{chunk_id, text, source, page, chunk_index}`. |
| 9 | Register | `register_document` | Stamps the `Document` node with filename, page/entity/relation counts. Invalidates the entity-linker cache. |

**One Document per PDF:** both entry points hash the file bytes, skip a PDF whose
sha256 is already a `Document`, and write + register each PDF as its own Document.
*Before this fork* `run_stage1.py --pdf-dir` loaded every chunk under **one**
Document named after the first PDF (and `--pdf` never registered a sha256). Because
`get_source_map` maps chunk sources to `Document.filename`, facts from the 2nd and 3rd
PDF were cited as the 1st. Found while ingesting for the evaluation.

**All-or-nothing:** extraction (the step that can hit rate limits) runs **before**
anything is written (for `--pdf-dir`, for all PDFs). If the quota runs out, `RateLimitExhausted` propagates, the API
returns 429 and Neo4j/Qdrant are untouched. Finished chunks are already cached, so
re-uploading resumes.

`run_stage2.py` + `enrichment/` optionally import PrimeKG edges (stage = `primekg`),
validated with the same ontology.

---

## 3. Query pipeline, step by step

All of this is `query()` in `backend/main.py`. Request model `QueryRequest`:
`question, provider, top_k=5, top_n=8, use_query_rewrite=True,
use_completeness_check=True, use_llm_cypher=False, force_no_gds=False,
min_confidence=0.5`.

| Step | Code | LLM call? | Detail |
|---|---|---|---|
| 0. Rewrite | `pipeline/query_rewriter.py` `rewrite_query` | yes (300 tok) | Fixes spelling, expands abbreviations, and **only if** the question has distinct parts returns 2-3 `sub_queries` (`MAX_SUB_QUERIES=3`). Any failure → `no_rewrite()` (pass-through). `search_queries` = rewrite + sub-queries, case-insensitively de-duplicated. The **answer** still responds to the original wording. |
| 1. Vector search | `pipeline/vector_retriever.py` `retrieve_vector_context`, merged by `_merge_round_robin` (`main.py:406`) | no | One Qdrant search (top_k) per search query. Results are **interleaved** (best of each sub-query, then second-best…) so one sub-query can't fill all the early slots that later steps look at. |
| 1b. Source map | `graph/document_registry.py` `get_source_map` | no | Maps temp-file chunk sources back to the uploaded filename for citations. |
| 2. Spot entities | `pipeline/entity_extractor_lite.py` `spot_entities_in_chunks` | yes (512 tok) | Names of entities in the first 6 retrieved chunks (600 chars each). Falls back to scispaCy if installed. This is "Qdrant-first": graph search starts from entities that are actually in relevant text. |
| 3. Plan | `pipeline/query_planner.py` `plan_query` | yes (400 tok) | Returns `route`, `entities_mentioned`, `confidence`, `target_relations`, `target_labels`. Gets the first 4 chunks (300 chars) as grounding. Relations are whitelisted (`_validate_relations`). Failure → `hybrid_multi` with confidence 0.3. |
| 4. Link | `pipeline/entity_linker.py` `link_entities` | no | Union of planner + spotted entities → Neo4j names. 3 tiers: exact/alias dict (1.0) → rapidfuzz `partial_ratio ≥ 82` → embedding cosine ≥ 0.78. The node index is cached for the process and invalidated after ingest/delete. |
| 5. Graph retrieval | `pipeline/graph_retriever.py` `retrieve_subgraph` (`:108`) | only if `use_llm_cypher` | Skipped for `vector_only`. Three strategies in priority order: (a) LLM-generated Cypher (opt-in), (b) **intent-aware** Cypher when the planner gave `target_relations` (`_intent_aware_subgraph`, `:252`), (c) **induced subgraph** 1 or 2 hops (`_induced_subgraph`, `:376`). Edges below `min_confidence` are dropped; temp-file sources are mapped to filenames; duplicates merged (`merge_duplicate_edges`). |
| 6. GDS | `pipeline/gds_retriever.py` `retrieve_gds_context` | no | Route-driven (see §4). Silently returns `[]` if the GDS plugin is missing. |
| 7. Rerank | `pipeline/reranker.py` `fuse_and_rerank` (`:43`) | no | Candidates = chunks + graph path strings + GDS strings. De-dup by first 100 chars, score with cross-encoder `BAAI/bge-reranker-v2-m3` (sigmoid → 0-1), `final = 0.4·original + 0.6·rerank`, keep `top_n`. If the model can't load, every rerank score is 0.5 (degrades to original ordering). |
| 8. Answer | `pipeline/generator.py` `generate_answer` | yes (1024 tok) | Context = structured subgraph block (`format_subgraph_for_context`, nodes + edges with `[file, p.N]` + evidence excerpts) prepended to the reranked items. Prompt rules: only use context, cite `[file.pdf, p.N]`, no graph jargon, say *"Insufficient information in the provided sources."*, end with "Sources used:". |
| 8b. Completeness *(fork)* | `pipeline/completeness.py` `check_coverage`, `main.py` `_repair_incomplete_answer` | yes (400 tok), +1 if repaired | Skipped for `vector_only` questions without sub-queries (`should_check`). The LLM splits the question into ≤ 4 parts (reusing the rewriter's sub-queries) and marks each covered / missing in the draft. For up to 2 missing parts: one vector search per part, the 2 best **new** chunks (de-duplicated, reranked against that part) are appended to the context, and the answer is regenerated **once** with the parts listed in the prompt (`required_parts`). Fails open: if the check errors, the draft is kept. Reported in the response as `completeness`. |
| 9. Response | end of `query()` | — | Builds `provenance` (documents de-duplicated by (source, page) keeping the max score; graph items carry `doc_source/doc_page/excerpt`), plus `retrieved_contexts` (exactly what the LLM saw, including repair chunks — this is what RAGAS scores). |

So a normal query costs **4 LLM calls** (rewrite, spot, plan, answer), 5 with LLM-Cypher,
plus 1 for the completeness check (2 when it repairs) on routes other than a plain
`vector_only`.

If Qdrant or Neo4j is unreachable, exception handlers return **503** with
`{"detail", "service"}` naming the store and the setting to check
(`VectorStoreUnavailable`, `neo4j.exceptions.ServiceUnavailable`, `AuthError`).
Before this fork it was a bare 500.

---

## 4. The four routes

Defined in `pipeline/query_planner.py` (`Route` literal + prompt at `:50`). The LLM
chooses; the code enforces what each route does:

| Route | When the planner should pick it | Graph retrieval (`main.py`) | GDS (`gds_retriever._route_to_mode`) |
|---|---|---|---|
| `vector_only` | definitions, mechanisms, general facts ("What is metformin?") | **skipped** (`main.py:229`) | skipped |
| `hybrid_single` | one hop from a named entity ("side effects of metformin") | intent-aware if relations given, else 1-hop induced subgraph | node similarity (Jaccard) for the first 2 entities |
| `hybrid_multi` | 2+ hops across entity types ("diabetes drugs that interact with BP drugs") | intent-aware, else **2-hop** induced subgraph | shortest path between the first 2 linked entities (needs ≥ 2) |
| `gds_augmented` | explicit path / similarity / centrality / ranking questions | induced subgraph (planner sets no target relations) | "full": shortest path + similarity + PageRank |

Planner failure defaults to `hybrid_multi` — the most thorough route — rather than the
cheapest one. Old route names (`no_graph`, `single_hop`, `multi_hop`) are mapped by
`_normalize_route`.

**What the evaluation shows:** the planner picked `hybrid_single` for 4 of the 5
multi-hop test questions (`evaluation/results/answers_*.json`) and the rewriter
produced sub-queries for only 2 of 13. Routing is conservative.

---

## 5. Keeping LLM-generated Cypher read-only

Only used when the request sets `use_llm_cypher=true`
(`pipeline/graph_retriever.py:150-176` → `pipeline/cypher_generator.py`). Defence in depth:

1. **Prompt** — told to write read-only queries starting with `MATCH`, with `LIMIT $limit`.
2. **Parameters, not string-building** — the LLM writes `$entity1..3` and `$limit`
   placeholders; actual entity names are passed as driver parameters, so a node named
   `"x'}) DETACH DELETE n //"` can't inject anything.
3. **Static validator** `_is_safe_cypher`: must start with `MATCH` /
   `OPTIONAL MATCH` / `WITH`; whole-word regex `_WRITE_KEYWORDS` rejects
   `CREATE DELETE DETACH MERGE SET REMOVE DROP FOREACH LOAD CSV CALL` (blocking `CALL`
   also blocks procedures such as `apoc.*` and `dbms.*`); and it must **end** with
   `LIMIT $param` or `LIMIT n`. `_inject_limit_if_missing` appends one when there is
   none and caps literal limits at 200 (`_MAX_LIMIT`).
4. **Database-enforced read transaction** — `Neo4jClient.run_read_query`
   (`graph/neo4j_client.py`) uses `session.execute_read`. Neo4j rejects any write
   inside a READ-access transaction, so even a query that slipped past the regex can't
   modify data. The work function carries a **server-side timeout**
   (`@unit_of_work(timeout=CYPHER_TIMEOUT_SECONDS)`, default 10 s), so a runaway
   `[*1..10]` path query is killed by Neo4j instead of hanging the request.
5. **Fallback** — if all candidates are unsafe/empty/fail, it falls through to the
   template strategies.

The non-LLM Cypher is safe for a different reason: the only values put into query text
are relation types and labels, and those come from whitelists
(`_validate_relations`; `NODE_LABELS` check in `/graph/explore`). Everything else is a
parameter.

**Before this fork:** no query timeout, and the `LIMIT` check was a substring test
(a `LIMIT` inside a subquery or string passed). **Still true:** the regex can reject
harmless queries that contain e.g. a property called `set`. Layer 4 is the real
guarantee; layer 3 is a fast first filter. `tests/test_cypher_guard.py` covers
both layers.

---

## 6. Rate limiter, retries and the per-chunk cache

### Proactive throttle — `llm/rate_limiter.py` `RateLimiter`
* One limiter **per provider**, created once (`_get_limiter`, `lru_cache`), shared by
  all threads (FastAPI runs sync endpoints in a thread pool) behind a `threading.Lock`.
* Sliding **60-second window** of `[timestamp, tokens]` events.
* `acquire(tokens)` (`:44`) computes the wait needed so that **both** RPM and TPM fit:
  * RPM: if `len(events) ≥ rpm`, wait until the event `rpm` positions from the end expires.
  * TPM: walk events oldest-first, "expiring" them until `used + tokens ≤ tpm`.
  * Also respects `_blocked_until` (set after a server 429).
  It sleeps **outside** the lock, then re-checks (loop) — so other threads aren't blocked.
* The reservation is an **estimate**: `(len(system)+len(user))/4 + max_tokens`
  (providers count `max_tokens` up front). After the call, `record_usage` replaces it
  with the real `usage.total_tokens`; `cancel` zeroes it if the request was rejected.

### Reactive handling — `llm/client.py` `chat()` (`:195`)
* `max_retries=0` on the OpenAI SDK so retries happen in one place.
* **429** → parse wait from the `retry-after` header or the message ("try again in
  1m26.4s", `_retry_after_seconds`), `limiter.block_for(wait)` pauses **all** callers,
  retry up to 6 times; otherwise exponential backoff 5, 10, 20… s (max 60).
* Gives up with `RateLimitExhausted` when: it's a daily quota (`_is_daily_quota`:
  "per day", "(TPD)"…), the wait exceeds `LLM_MAX_WAIT_SECONDS` (120), retries are used
  up, or "request too large" (waiting can't help).
* Connection/timeout/5xx → 3 retries with 2, 4, 8 s backoff.
* `RateLimitExhausted` is mapped to **HTTP 429** with `retry_after_seconds`
  (`backend/main.py:72`); provider 401/404 → **502** with a hint (`:89`).
* Reasoning models on Groq (`gpt-oss`, `qwen3`) get `reasoning_effort` low/none and
  +1024 output tokens of headroom, because hidden thinking counts against `max_tokens`.

### Per-chunk extraction cache — `ingestion/entity_extractor.py`
* Key = hash of the chunk text + a **fingerprint** (`_cache_fingerprint`): sha256 of
  the extraction model, system and user prompts, entity/relation types, the ontology
  and the confidence thresholds, truncated to 12 chars. Value = validated entities +
  relations (and the fingerprint, for inspection).
* Content-addressed: the same text in a re-upload, or in another PDF, is never paid for
  twice — as long as the extraction setup is unchanged.
* Combined with all-or-nothing writes, an interrupted 200-chunk ingestion resumes from
  the first uncached chunk.
* **Before this fork** the key was only `sha256(chunk_text)`, so after changing
  `EXTRACTION_MODEL`, the prompt or a threshold you silently got the old extractions
  unless you deleted `.cache/` yourself. Now a change produces new keys. One-time
  cost: caches written by the original code are not reused (they're re-extracted once).

---

## 7. Graph data model

```
(:Document {sha256, filename, ingested_at, page_count, entity_count, relation_count})
      │ HAS_CHUNK
      ▼
(:Chunk {chunk_id, text[:2000], source, page, chunk_index, token_count})
      │ MENTIONS {confidence}
      ▼
(:Drug|:Disease|… {name, aliases, confidence, pages, chunk_ids, sources, stage,
                   extracted_at, umls_cui, mesh_id, rxnorm_id})
      │ TREATS|CAUSES|… {confidence, page_numbers, chunk_ids, sources, paragraph_ids,
      │                  stage, extracted_at, valid_from, valid_until}
      ▼
(:Entity)

(:Entity)-[:HAS_EVIDENCE {predicate}]->(:Evidence {evidence_id, subject, predicate,
                                                    object, excerpt, confidence})
(:Evidence)-[:FROM_CHUNK]->(:Chunk)
(:Evidence)-[:FROM_DOCUMENT]->(:Document)
```

* **15 entity labels** (`graph/neo4j_client.py:29`): Disease, Drug, Symptom, Treatment,
  SideEffect, PatientGroup, Organ, MedicalGuideline, Gene, Protein, Pathway, Compound,
  CellType, MolecularFunction, BiologicalProcess.
* **16 relation types** (`:39`): TREATS, CAUSES, INTERACTS_WITH, HAS_SYMPTOM,
  CONTRAINDICATED_FOR, AFFECTS_ORGAN, RECOMMENDED_BY, TARGETS, REGULATES, INHIBITS,
  ACTIVATES, LEADS_TO, BIOMARKER_FOR, EXPRESSED_IN, PARTICIPATES_IN, RELATED_TO.
* Entity identity is `(label, name)` via `MERGE`. Because every PDF writes into the same
  nodes, "Metformin" from the diabetes guide and "Metformin" from the interactions
  reference are **one node**. That's what makes cross-document multi-hop work.
* **Evidence nodes** make provenance first-class: one per (subject, predicate, object,
  chunk), id = `sha256(s|p|o|chunk_id)[:16]` → idempotent re-ingestion. The excerpt is
  the first sentence of the chunk mentioning the subject or object
  (`_best_excerpt`). The intent-aware query joins Evidence in the same Cypher, which is
  how graph facts get `[file, p.N]` citations plus a supporting sentence.
* **Temp-file sources:** API uploads are parsed from a temp file, so `Chunk.source`
  and the Qdrant payload hold the temp name. `get_source_map` maps it back to
  `Document.filename` at query time; delete uses the chunk sources to purge Qdrant.
* **Provenance accumulates across documents** (`graph/neo4j_client.py` `_union`,
  `_max_conf`): the upserts union `aliases`, `sources`, `chunk_ids`, `pages`,
  `page_numbers` and `paragraph_ids` with plain Cypher
  (`reduce(acc = [], x IN old + new | CASE WHEN x IN acc THEN acc ELSE acc + x END)`),
  so no APOC is needed; confidence is the **max** over documents.
  **Before this fork** (bug found in the audit) the batch upserts did
  `SET n.sources = row.sources` etc., so ingesting a second PDF **replaced** the first
  PDF's provenance on shared nodes and edges, and confidence was `(old + new) / 2`,
  weighted towards the latest document. Evidence nodes were unaffected (they're per
  chunk), which is why intent-aware citations still worked.
* **Deleting a document** (`graph/document_registry.py` `delete_document`) now also
  removes its provenance: Evidence nodes from its chunks are deleted, and its
  chunk ids and sources are stripped from every shared node and edge. Entities and
  edges themselves are kept, because on graphs built by the old code their lists are
  incomplete, so "no sources left" doesn't reliably mean "no document supports this".
  `tests/test_provenance.py` checks this against a live Neo4j.

---

## 8. Evaluation

`evaluation/run_ragas.py`: 13 questions (`testset.json`; 3 definition, 4 single-hop,
5 multi-hop, 1 unanswerable; 8 clean, 5 messy). Step 1 posts each question to the
in-process app (`TestClient`) twice (rewrite off/on) and caches answers per question.
Step 2 judges with RAGAS (faithfulness, answer relevancy, context recall, answer
correctness). Judge calls go through the same rate-limited client, metrics are split
across two Groq models to fit per-model daily quotas, and each metric always uses the
same judge so configs stay comparable. NaN metrics are retried on the next run.

Multi-hop answer correctness from the cached scores: **0.72** without rewriting,
**0.65** with rewriting (Q8 0.63→0.49, Q12 0.72→0.46). Context recall is 1.0 on all of
them. So the facts **were retrieved**, but the answer left parts out. That's the
"answer completeness" weakness the README describes.

**Fork measurement of the completeness check** (`evaluation/results/fork/`, fresh
ingestion, rewrite on, check off → on, same judges):

| Multi-hop, 5 questions | off | on |
|---|---|---|
| Faithfulness | 0.80 | 1.00 |
| Answer correctness | 0.68 | 0.72 |
| Answer correctness without Q12 | 0.72 | 0.70 |

* Only **Q8** was flagged and repaired (0.56 → 0.58). The checker correctly saw that
  insulin/glipizide–metoprolol was missing, but that fact was **already in the
  context** (chunks 1–2), so re-retrieving for the part brought back the same
  chunks and the regenerated answer still left it out.
* Q9–Q12 weren't flagged; their differences are sampling noise. Q12's "off" answer
  degenerated into ~1,800 zero-width spaces (a one-off model glitch), which is most of
  the +0.04.
* Conclusion: detection works, repair-by-retrieval doesn't help when the problem is
  generation. Next: give the generator the specific missing item and the chunk that
  holds it, or have the checker cite the supporting chunk. Run this on more than
  5 questions before trusting any ±0.05.

---

## 9. Other things found in the audit

* *(Fixed in this fork)* No tests: `test_connection.py` and `test_get_source_map.py`
  are scripts that need a live Neo4j (`pytest` collected 0 tests), and no CI. Now
  `tests/` has 119 tests and `.github/workflows/ci.yml` runs them.
* *(Fixed in this fork)* `run_stage1.py --pdf-dir` put all PDFs under one Document,
  so citations named the wrong file (see §2).
* `requirements.txt` uses only `>=` lower bounds, so a fresh install on 2026-11-18
  resolved fastapi 0.163 / torch 2.15 / ragas 0.4.4. Builds are not reproducible.
  (The fork pins only the test dependencies, in `requirements-dev.txt`.)
* *(Fixed in this fork)* If Qdrant or Neo4j was down, `POST /query` returned a bare
  `500 Internal Server Error` instead of a clear 503.
* *(Fixed in this fork)* The rewriter's sub-query validator ran after Pydantic's type
  check, so one non-string item in `sub_queries` failed validation and threw away the
  whole rewrite. It now runs in `mode="before"` and drops only the bad item.
* No LICENSE file upstream.
* Unauthenticated destructive endpoints (`DELETE /documents/{sha}`,
  `POST /documents/purge-legacy`) and `allow_origins=["*"]`. Fine for a local demo,
  not for deployment.
* Entity linker tier 2 uses `partial_ratio`, so short terms ("ACE", "AF") can fuzzy-match
  any longer name that contains them.
* GDS uses one fixed projection name (`rag_live`) for every request, so two concurrent
  GDS queries can drop each other's projection.
* `get_source_map` scans every Document→Chunk pair on every query.
* `EMBED_MODEL` defaults to BGE in `ingestion/embedder.py` but to MiniLM in
  `pipeline/entity_linker.py`; setting the variable changes both.
* `backend/main.py:404`: a comment about `/ingest` running in a worker thread sits above
  `_merge_round_robin` instead of `ingest()`.
* The reranker scores graph triples written as `A --[TREATS]--> B` with a cross-encoder
  trained on natural-language passages, and mixes Qdrant cosine scores with graph
  confidences on the same 0.4 weight.

---

## 10. Likely interview questions (answers based on the code)

1. **Why graph + vectors instead of plain RAG?**
   Vector search finds passages similar to the question, but a cross-document question
   ("diabetes drugs that interact with BP drugs") needs facts from two PDFs joined by a
   shared entity. In the graph both PDFs write into the same `Drug {name:"Metformin"}`
   node, so one Cypher traversal connects them. Vectors still cover definitions and
   nuance that triples lose. The reranker fuses both.

2. **Walk me through a query.**
   Rewrite → one Qdrant search per sub-query, round-robin merged → LLM spots entities
   in the top chunks → planner picks a route plus target relations → entities linked to
   Neo4j names → intent-aware Cypher (with Evidence join) or induced subgraph, plus GDS
   → cross-encoder reranks chunks and paths together → answer LLM with citations. Four
   LLM calls.

3. **How does the planner decide the route, and what if it's wrong?**
   An LLM classifies the question into one of four routes with few-shot examples and
   returns target relation types. Pydantic validates the route; relations are
   whitelisted. If parsing fails it falls back to `hybrid_multi` at confidence 0.3, the
   most thorough route. A wrong `vector_only` is the costly mistake, because graph
   retrieval is skipped entirely.

4. **How do you stop the LLM's Cypher from deleting the database?**
   Four layers: prompt, entity names passed as parameters, a regex validator that
   whitelists the first clause and rejects write keywords and `CALL`, and execution in
   a Neo4j READ transaction (`execute_read`), where the database itself refuses writes.
   The last one is the real guarantee. It's also opt-in (`use_llm_cypher`). Against
   expensive reads: every query must end in a `LIMIT` (capped at 200) and runs with a
   10 s server-side timeout.

5. **Isn't building Cypher with f-strings an injection risk?**
   Only relation types and labels go into the query text, and both come from fixed
   whitelists (`_validate_relations`, `NODE_LABELS`). All user- or LLM-derived values
   (names, limits, label filters) are driver parameters.

6. **How do you prevent hallucinated entities in the graph?**
   `_is_grounded`: an entity is kept only if its name, an alias, or all of its
   significant words appear in the chunk text. There are also confidence thresholds
   (0.40 entity, 0.50 relation) and an ontology domain/range check per predicate.
   Caveat: relations aren't grounded separately, so a relation between two grounded
   entities can still be hallucinated.

7. **How do citations get to file and page level?**
   Chunks never cross pages, so each has one page. Each relation gets Evidence nodes
   linked to the chunk it came from. The intent-aware query joins
   `(seed)-[:HAS_EVIDENCE]->(ev)-[:FROM_CHUNK]->(c)` and returns `c.source, c.page,
   ev.excerpt` per edge. Temp-file names are mapped back to the real filename with
   `get_source_map`.

8. **How does the rate limiter work?**
   It's a sliding 60-second window of (time, tokens) events, one per provider, shared
   across threads. Before each call it reserves the estimated tokens (prompt/4 +
   max_tokens) and sleeps until both RPM and TPM fit, then corrects the reservation with
   real usage afterwards. A 429 from the server pauses every caller for the
   `retry-after` time. A daily quota raises `RateLimitExhausted`, which becomes HTTP 429.

9. **What happens if the quota runs out halfway through a 200-page PDF?**
   Extraction runs before any database write, so nothing is written. Each finished chunk
   is cached by the hash of its text, so re-uploading skips straight to the first
   uncached chunk. The API returns 429 with the retry time.

10. **Weakness of that cache?**
    Originally the key was only the chunk text, so changing the model or prompt kept
    serving old extractions. I added a fingerprint of everything that affects the
    output (model, prompts, types, ontology, thresholds) to the key. Hashing the inputs
    instead of a hand-maintained version number means nobody has to remember to bump it.

11. **What does query rewriting buy you, measured?**
    On messy questions, context recall went 0.93 → 1.00 and answer correctness
    0.79 → 0.84. On the multi-hop subset, answer correctness dropped 0.72 → 0.65. The
    rewrite helps retrieval, but the answer then misses parts of multi-part questions.

12. **Why rerank with a cross-encoder after vector search?**
    A bi-encoder (BGE) embeds the query and the chunk separately: fast, but coarse. A
    cross-encoder reads (query, candidate) together and scores relevance much more
    accurately, but it's too slow for the whole corpus, so it runs on the ~20 candidates
    only. It also puts graph paths and chunks on one 0-1 scale so they can be ranked
    together (final = 0.4·original + 0.6·rerank).

13. **How is entity resolution handled at ingestion and at query time?**
    At ingestion: normalisation plus an abbreviation map, alias merge, then rapidfuzz
    `ratio ≥ 90`. At query time: exact/alias lookup, then `partial_ratio ≥ 82`, then
    embedding cosine ≥ 0.78. Weakness: `partial_ratio` over-matches short acronyms.

14. **How do you evaluate it, and how much do you trust the numbers?**
    RAGAS with LLM judges, cached per question, each metric pinned to one judge model.
    But it's only 13 questions (5 multi-hop), one run, LLM-judged. A ±0.1 change on one
    question moves the multi-hop mean by 0.02. Treat small differences as noise. What
    the data supports clearly: retrieval is near-complete (recall ≈ 1.0), and
    completeness of the answer is the bottleneck.

15. **What would you change for production?**
    Already done in the fork: query timeouts, provenance that accumulates across
    documents, a model-aware cache key, 503s for a missing store, tests + CI, Docker
    Compose. Still to do: auth on the destructive endpoints and a CORS allow-list;
    per-request GDS projection names; streaming answers; ingestion as a background job
    instead of a blocking request; pinned runtime dependencies; a larger, versioned
    eval set run in CI.

16. **How does the completeness check work, and why that design?**
    The eval showed recall ≈ 1.0 but low answer correctness on multi-hop questions:
    the facts were retrieved, then a part of the question was dropped. So after the
    draft, one cheap LLM call lists the question's parts and marks which the answer
    covers. For up to two missing parts it does a targeted vector search, adds the two
    best new chunks per part, and regenerates once with the parts spelled out. It's
    bounded (no loop, at most 2 extra calls), skipped for simple definition questions,
    fails open, and reports what it did in the response so it can be audited.
    Measured result, honestly: it flagged the one incomplete answer (Q8) correctly,
    but the repair didn't fix it, because the missing fact was already in the context.
    Answer correctness didn't measurably change. That told me the bottleneck is
    generation, not retrieval, which is a useful finding in itself.

17. **How did you test code that calls an LLM and two databases?**
    The LLM is replaced by a `FakeLLM` fixture that returns canned JSON and records
    prompts; the API tests use FastAPI's `TestClient` with retrieval functions
    monkeypatched; the rate limiter uses a fake clock. Cypher upserts are tested by
    capturing the generated query, plus one integration test against a real Neo4j
    (CI starts a Neo4j service container) that ingests two documents, checks the
    provenance is unioned, and deletes one.

---

## 11. Changes in this fork (one line each, oldest first)

| Commit subject | Files | Why |
|---|---|---|
| Ignore local virtualenv and test caches | `.gitignore` | Keep `.venv/`, `.pytest_cache/` out of git; un-ignore `.env.example` |
| Add architecture notes | `docs/ARCHITECTURE_NOTES.md` | This file |
| Fix sub-query cleaning | `pipeline/query_rewriter.py` | One bad item discarded the whole rewrite (found by the new tests) |
| Add unit tests | `tests/`, `pytest.ini`, `requirements-dev.txt` | Fake LLM, fake clock, `TestClient`; no network |
| Add GitHub Actions workflow | `.github/workflows/ci.yml` | Tests with a Neo4j service, frontend lint + build, Docker build |
| Accumulate provenance across documents | `graph/neo4j_client.py`, `graph/document_registry.py` | Second PDF overwrote the first's sources; delete now removes only its own provenance |
| Return 503 when a store is down | `backend/main.py`, `vector/qdrant_client.py` | Bare 500 → 503 naming Qdrant / Neo4j |
| Trailing LIMIT + timeout for generated Cypher | `pipeline/cypher_generator.py`, `graph/neo4j_client.py` | Bound the cost of a read the LLM wrote |
| Key the extraction cache on model and prompt | `ingestion/entity_extractor.py` | Stale extractions after a model/prompt change |
| Docker Compose | `Dockerfile`, `docker-compose.yml`, `frontend/Dockerfile`, `frontend/nginx.conf`, `.env.example` | One-command setup |
| Answer completeness check | `pipeline/completeness.py`, `pipeline/generator.py`, `backend/main.py` | Multi-hop answers dropped parts despite recall 1.0 (§3 step 8b) |
| RAGAS runner targets configs / subsets | `evaluation/run_ragas.py`, `evaluation/compare.py` | Measure only what changed; keep the original results untouched |
| One Document per PDF in the CLI | `run_stage1.py` | `--pdf-dir` cited facts from every PDF as the first PDF |
| Measured completeness check, README | `evaluation/results/fork/`, `README.md`, this file | Before/after numbers (§8) |
