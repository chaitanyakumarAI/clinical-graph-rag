# Medical Graph RAG for Clinical Reasoning

A hybrid **knowledge graph + vector search** RAG system for answering medical questions
from your own PDFs. It extracts drugs, diseases, symptoms, genes and their relationships
into a Neo4j knowledge graph, stores text chunks in Qdrant, and answers questions with
citations to the exact file and page.

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

### Prerequisites

- Python 3.11+ and Node.js 18+
- **Neo4j 5** on `bolt://localhost:7687` (the Graph Data Science plugin is optional)
- **Qdrant** on `http://localhost:6333`, e.g. `docker run -p 6333:6333 qdrant/qdrant`
- An LLM API key: Groq, Gemini or OpenAI
### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Configure `.env` (project root)

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

### 3. Run

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

If the LLM quota runs out mid-run, nothing is written. Re-running the same command later
resumes from the cache in `.cache/extractions/`.

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
evaluation/                RAGAS runner, test set, report generator, results
frontend/                  React + Vite UI (query, upload, graph explorer, stats)
data/pdfs/                 Sample medical PDFs and test questions
run_stage1.py, run_stage2.py   CLI ingestion and enrichment
STARTUP.md                 Detailed setup, rate-limit tuning and troubleshooting
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `429` / rate limit during ingestion | Quota used up; retry later (progress is cached) or enable **Large PDF mode** |
| `LLM provider error (401)` | API key invalid or revoked |
| `LLM provider error (404)` | Model not available on your account; check `<PROVIDER>_MODEL` |
| `Connection refused :7687` / `:6333` | Start Neo4j / Qdrant |
| Graph Explorer is empty | Ingest a PDF first |

More in [STARTUP.md](STARTUP.md#troubleshooting).

---

> **Disclaimer:** this is a research and educational project. Its answers are not medical
> advice and must not be used for clinical decisions.
