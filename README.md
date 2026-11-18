# Medical Graph RAG for Clinical Reasoning

A hybrid **knowledge graph + vector search** RAG system for answering medical questions
from your own PDFs. It extracts drugs, diseases, symptoms, genes and their relationships
into a Neo4j knowledge graph, stores text chunks in Qdrant, and answers questions with
citations to the exact file and page.

> **About this fork.** The original project is by
> [DongaShanmukhaSivaVenkataSai](https://github.com/DongaShanmukhaSivaVenkataSai/Medical-Graph-RAG-for-Clinical-Reasoning-),
> whose commits are kept unchanged at the start of this history. This fork adds tests
> and CI, Docker Compose, several correctness fixes and an answer-completeness check;
> see [Changes in this fork](#changes-in-this-fork).

Because facts from every PDF land in one shared graph, the system can answer questions
whose answer is spread across several documents. For example, *"Which diabetes drugs
interact with blood-pressure medications?"* needs the drug list from one PDF and the
interactions from another.

```
          ┌────────────── Ingestion ──────────────┐
PDF ──► parse ──► chunk ──► LLM entity/relation extraction ──► deduplicate ──► Neo4j graph
                    └──► embed (BGE) ───────────────────────────────────────► Qdrant vectors

          ┌──────────────────────────────── Query ────────────────────────────────┐
Question ──► rewrite ──► vector search ──► spot entities ──► plan route ──► graph retrieval
                                                                                │
             answer with citations ◄── LLM ◄── rerank (cross-encoder) ◄─────────┘
```

---

## Features

| Area | What it does |
|---|---|
| **Ingestion** | PDF → chunks → LLM extraction of 15 entity types and 16 relation types, with grounding checks (an entity must appear in the text) and ontology validation of every triple |
| **Provenance** | Every node and edge records its source document, page and chunk; `Evidence` nodes link each relation to the sentence that supports it |
| **Query rewriting** | Fixes typos, expands abbreviations (*T2DM*, *BP*, *AF*) and splits multi-part questions into sub-queries before retrieval |
| **Adaptive routing** | An LLM planner picks a route per question: `vector_only`, `hybrid_single`, `hybrid_multi` or `gds_augmented` |
| **Hybrid retrieval** | Qdrant semantic search + intent-aware Cypher (only the relation types the question asks about) + optional LLM-generated Cypher, run in a read-only transaction |
| **Reranking** | BGE cross-encoder fuses graph paths and document chunks into one ranked context |
| **Grounded answers** | Answers cite `[file.pdf, p.N]`, and say *"Insufficient information"* when the documents don't cover the question |
| **Rate-limit safe** | One shared LLM client paces calls under the provider's per-minute limits, honours `retry-after`, and stops cleanly when the daily quota runs out; a per-chunk cache lets long PDFs resume |
| **Frontend** | React UI: ask questions, upload PDFs, explore the graph (Cytoscape) and view statistics |
| **Evaluation** | RAGAS pipeline comparing configurations, producing a PDF report |

---

## Quick start

### Option A: Docker Compose (one command)

Needs Docker with Compose v2. Starts Neo4j, Qdrant, the API and the UI.

```bash
cp .env.example .env      # set NEO4J_PASSWORD (8+ chars) and one LLM key, e.g. GROQ_API_KEY
docker compose up --build
```

| Service | URL |
|---|---|
| UI | http://localhost:8080 |
| API docs | http://localhost:8000/docs |
| Neo4j browser | http://localhost:7474 (user `neo4j`, your `NEO4J_PASSWORD`) |

The backend image uses CPU-only PyTorch. The embedding and reranker models (~3 GB) are
downloaded on first use into the `hf-models` volume, so the first query is slow and
later ones aren't. The extraction cache lives in the `extraction-cache` volume, so
re-ingesting a PDF costs no LLM calls. Data persists in the `neo4j-data` and
`qdrant-data` volumes. `docker compose down -v` wipes everything.

To enable the optional GDS route, add `NEO4J_PLUGINS=["graph-data-science"]` to `.env`.
To use a local Ollama model, set `LLM_PROVIDER=ollama`; the backend reaches the host's
Ollama at `host.docker.internal`.

### Option B: run locally

#### Prerequisites

- Python 3.11+ and Node.js 18+
- **Neo4j 5** on `bolt://localhost:7687` (the Graph Data Science plugin is optional)
- **Qdrant** on `http://localhost:6333`, e.g. `docker run -p 6333:6333 qdrant/qdrant`
- An LLM API key: Groq, Gemini or OpenAI

#### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

#### 2. Configure `.env` (project root)

Copy `.env.example` to `.env`; it lists every setting. The minimum:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
QDRANT_URL=http://localhost:6333

# One LLM provider (auto-detected in the order Gemini → Groq → OpenAI)
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TPM=8000            # your tokens/minute limit (console.groq.com/settings/limits)
```

#### 3. Run

```bash
# Terminal 1 — backend (project root)
uvicorn backend.main:app --port 8000 --reload

# Terminal 2 — frontend
cd frontend && npm run dev
```

Open **http://localhost:5173**, upload a PDF in **Upload PDF**, then ask a question in
**Ask Question**. Interactive API docs are at http://localhost:8000/docs.

Sample documents with known answers are in [`data/pdfs/`](data/pdfs/), with test questions
in [`data/pdfs/TEST_QUESTIONS.md`](data/pdfs/TEST_QUESTIONS.md).

---

## Configuration

All settings live in `.env`. Only the database URLs and one LLM key are required.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | auto | Force `gemini`, `groq`, `openai` or `ollama` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` | — | Provider credentials |
| `<PROVIDER>_MODEL` | see `llm/client.py` | Model name, e.g. `GROQ_MODEL=openai/gpt-oss-120b` |
| `<PROVIDER>_TPM` / `<PROVIDER>_RPM` | free-tier values | Tokens / requests per minute the client paces to |
| `EXTRACTION_MODEL` | provider default | Separate model just for bulk PDF extraction |
| `EXTRACTION_MAX_TOKENS` | `2048` | Output budget per chunk |
| `LLM_REASONING_EFFORT` | `low` (gpt-oss), `none` (qwen3) | Thinking budget for reasoning models |
| `LLM_MAX_WAIT_SECONDS` | `120` | Longest wait on a 429 before giving up |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `localhost:11434/v1`, `llama3.1:8b` | Local model, no rate limits |
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Chunk embeddings |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `CYPHER_TIMEOUT_SECONDS` | `10` | Server-side timeout for LLM-generated Cypher |
| `QDRANT_COLLECTION` | `medical_chunks` | Vector collection name |
| `UMLS_API_KEY` | — | Optional UMLS identifier linking |

Long PDFs on free-tier APIs: see **Rate limits** in [STARTUP.md](STARTUP.md).

---

## Usage

### Ingest from the command line

```bash
python run_stage1.py --pdf data/pdfs/diabetes_management_guide.pdf
python run_stage1.py --pdf-dir data/pdfs/ --chunk-tokens 1024   # fewer LLM calls for long PDFs
python run_stage1.py --pdf my.pdf --dry-run --output-json out.json
```

Each PDF becomes its own document (identified by the SHA-256 of the file), and PDFs that
are already ingested are skipped. If the LLM quota runs out mid-run, nothing is written.
Re-running the same command later resumes from the cache in `.cache/extractions/`.

Optional Stage 2 enriches the graph with [PrimeKG](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM):

```bash
python run_stage2.py --nodes data/seed/primekg_nodes.csv --edges data/seed/primekg_edges.csv
```

### API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Ask a question; returns the answer, route, rewritten query, graph paths, provenance and contexts |
| `POST` | `/ingest` | Upload a PDF (`chunk_tokens`, `provider`, `link_ids` query parameters) |
| `GET` | `/documents` | List ingested documents |
| `DELETE` | `/documents/{sha256}` | Remove a document from Neo4j and Qdrant |
| `GET` | `/graph/explore` · `/graph/subgraph` · `/graph/node/{name}` · `/graph/stats` | Graph exploration |
| `GET` | `/health` | Health check |

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "side efects of metformn?", "use_query_rewrite": true}'
```

`/query` also accepts `use_completeness_check` (default `true`); the response's
`completeness` field shows the parts it found, which were missing, and whether the
answer was repaired. If Qdrant or Neo4j is unreachable, the API returns `503` with
the name of the store and the setting to check.

---

## Tests

```bash
pip install -r requirements-dev.txt     # light: no torch or models needed
pytest                                   # 118 unit/API tests, a few seconds, no LLM or database calls
```

The LLM is replaced by a fake that returns canned JSON, and the API tests run the real
`/query` handler with retrieval stubbed out. One integration test runs against a real
Neo4j when you point it at one (it only touches nodes it creates):

```bash
NEO4J_TEST_URI=bolt://localhost:7687 NEO4J_TEST_PASSWORD=... pytest -k neo4j
```

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs all tests
against a Neo4j service container, lints and builds the frontend, and builds the Docker
images.

---

## Evaluation

The pipeline was evaluated with [RAGAS](https://docs.ragas.io) on 13 questions (definitions,
single-hop and cross-document multi-hop questions, plus one that isn't answerable from the
documents), with and without query rewriting.

| Metric (12 answerable questions) | Without rewriting | With rewriting |
|---|---|---|
| Faithfulness | 0.94 | 0.92 |
| Answer relevancy | 0.85 | 0.86 |
| Context recall | 0.97 | **1.00** |
| Answer correctness | 0.76 | 0.75 |

On messy questions (typos, abbreviations) rewriting raised context recall from 0.93 to
**1.00** and answer correctness from 0.79 to **0.84**. Retrieval is near-complete; the main
remaining weakness is answer completeness on cross-document questions. The full report,
with per-question scores and recommendations, is
[`evaluation/RAGAS_Evaluation_Report.pdf`](evaluation/RAGAS_Evaluation_Report.pdf).

Re-run it:

```bash
python evaluation/run_ragas.py     # answer + judge (cached per question, resumable)
python evaluation/make_report.py   # rebuild the PDF
```

---

## Changes in this fork

Everything below was added on top of the original commits; each item is one or two
focused commits. Design details and trade-offs are in
[`docs/ARCHITECTURE_NOTES.md`](docs/ARCHITECTURE_NOTES.md).

| Change | Before | After |
|---|---|---|
| **Tests + CI** | `pytest` collected 0 tests; no CI | 119 tests (fake LLM, API tests, fake-clock rate limiter, live Neo4j test); GitHub Actions runs them, lints and builds the frontend, and builds the Docker images |
| **Provenance across documents** (bug) | Ingesting a second PDF **overwrote** `sources` / `chunk_ids` / `pages` on shared nodes and edges; confidence drifted towards the latest PDF | Lists are unioned (plain Cypher, no APOC); confidence is the max; deleting a document removes only its provenance and Evidence |
| **One Document per PDF in the CLI** (bug) | `run_stage1.py --pdf-dir` put all chunks under one Document named after the first PDF, so facts from the other PDFs were **cited as the wrong file** | Each PDF gets its own Document (sha256 + filename) like `POST /ingest`; already-ingested PDFs are skipped |
| **Sub-query cleaning** (bug) | One non-string sub-query from the LLM failed validation and discarded the whole rewrite | Bad items are dropped, the rest are kept |
| **Clear errors when a store is down** | `500 Internal Server Error` | `503` with `{"detail", "service"}` naming Qdrant or Neo4j and the setting to check |
| **Generated-Cypher limits** | `LIMIT` checked by substring; no timeout | Query must end in a `LIMIT` (capped at 200); 10 s server-side timeout (`CYPHER_TIMEOUT_SECONDS`) |
| **Extraction cache key** | Key was the chunk text only, so changing the model or prompt silently reused old extractions | Key includes a fingerprint of model, prompts, types, ontology and thresholds. Caches from the original code are re-extracted once |
| **Docker Compose** | Manual Neo4j / Qdrant / Python / Node setup | `docker compose up --build` (CPU-only torch, nginx-served UI, persistent volumes) |
| **Answer completeness check** | Multi-hop answers dropped parts of the question even though context recall was 1.0 | After the draft, an LLM check lists the question's parts; for up to 2 missing parts it retrieves more and regenerates once. +1 LLM call (+2 when repairing). Measured: detects the gap, no measurable gain yet (below) |

### Measured effect of the completeness check

The 3 sample PDFs were freshly ingested and the 5 multi-hop questions were answered
with query rewriting on, first without and then with the check (same judges as the
original evaluation). Results are in [`evaluation/results/fork/`](evaluation/results/fork/);
the original author's results are untouched.

| Multi-hop (5 questions) | Check off | Check on | Δ |
|---|---|---|---|
| Faithfulness | 0.80 | 1.00 | +0.20 |
| Answer relevancy | 0.84 | 0.87 | +0.03 |
| Context recall | 1.00 | 1.00 | 0 |
| Answer correctness | 0.68 | 0.72 | +0.04 |
| *Answer correctness without Q12* | *0.72* | *0.70* | *−0.02* |

**Honest reading: no measurable improvement in answer correctness.**

- The check changed an answer only for **Q8**. It correctly found that the answer listed
  the empagliflozin–diuretic interaction but left out insulin/glipizide–metoprolol, and
  it ran the repair (answer correctness 0.56 → 0.58).
- The repair didn't add the missing fact, because that fact was **already in the
  context** (chunks 1–2). The bottleneck is generation, not retrieval, and a second
  search for the same part returns the same chunks.
- For Q9–Q12 nothing was flagged, so differences there are run-to-run noise. Most of
  the +0.04 comes from Q12, where the "off" answer happened to degenerate into ~1,800
  zero-width characters, a one-off model glitch unrelated to the check.
- Cost: +1 LLM call per checked question, +2 when repairing (Q8: 84 s → 180 s under
  free-tier rate limits).

With 5 questions and one run each, differences of ±0.05 are within noise. What this
shows: the coverage check **detects** an incomplete answer, but a repair that
re-retrieves isn't enough. The next step would be to pass the generator the *specific*
missing item ("list every diabetes drug that interacts with a BP drug") instead of
re-searching. Total spend for this measurement: ingestion ≈ 40k tokens, answers 76k
(gpt-oss-120b), judging 60k (gpt-oss-20b) + 77k (qwen3).

---

## Project structure

```
backend/main.py            FastAPI app — /query pipeline, /ingest, graph and document endpoints
llm/                       Shared LLM client: providers, rate limiter, retries, usage tracking
ingestion/                 PDF parsing, chunking, embedding, LLM extraction, dedup, ID linking
pipeline/                  Query rewriter, planner, entity spotter/linker, graph/vector/GDS
                           retrieval, Cypher generator, reranker, answer generator
graph/                     Neo4j client, graph loader, evidence writer, document registry, GDS
vector/                    Qdrant wrapper
enrichment/                Stage 2 PrimeKG matching and edge import
evaluation/                RAGAS runner, comparison script, test set, report generator, results
frontend/                  React + Vite UI (query, upload, graph explorer, stats)
data/pdfs/                 Sample medical PDFs and test questions
tests/                     pytest suite (fake LLM, API tests, optional live Neo4j test)
docs/ARCHITECTURE_NOTES.md How the pipeline works, step by step, with design trade-offs
run_stage1.py, run_stage2.py   CLI ingestion and enrichment
Dockerfile, docker-compose.yml, frontend/Dockerfile   Container setup
STARTUP.md                 Detailed setup, rate-limit tuning and troubleshooting
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `429` / rate limit during ingestion | Quota used up; retry later (progress is cached) or enable **Large PDF mode** |
| `LLM provider error (401)` | API key invalid or revoked |
| `LLM provider error (404)` | Model not available on your account; check `<PROVIDER>_MODEL` |
| `503` naming Qdrant or Neo4j / `Connection refused :7687` / `:6333` | Start Neo4j / Qdrant, or check `NEO4J_URI` / `QDRANT_URL` / the Neo4j password |
| Graph Explorer is empty | Ingest a PDF first |

More in [STARTUP.md](STARTUP.md#troubleshooting).

---

> **Disclaimer:** this is a research and educational project. Its answers are not medical
> advice and must not be used for clinical decisions.
