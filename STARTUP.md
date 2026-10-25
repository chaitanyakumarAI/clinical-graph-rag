# Medical Graph RAG — Startup Guide

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│  Frontend (React + Vite)         :5173                 │
│  ├─ QueryPanel   — Multi-hop Q&A with provenance       │
│  ├─ UploadPanel  — PDF ingestion + UMLS/MeSH linking   │
│  ├─ GraphExplorer — Cytoscape.js interactive graph     │
│  └─ StatsDashboard — Real-time node/edge analytics     │
└──────────────────────┬─────────────────────────────────┘
                       │  HTTP (proxied in dev)
┌──────────────────────▼─────────────────────────────────┐
│  Backend (FastAPI + Uvicorn)     :8000                  │
│  ├─ POST /query   — Hybrid graph+vector retrieval       │
│  ├─ POST /ingest  — PDF → entities → Neo4j + Qdrant    │
│  ├─ GET  /graph/* — Graph exploration endpoints         │
│  ├─ GET  /documents — Document registry                 │
│  └─ GET  /health  — Health check                        │
└──────┬───────────────────────────┬─────────────────────┘
       │                           │
  ┌────▼──────┐             ┌──────▼──────┐
  │  Neo4j    │             │  Qdrant     │
  │  :7687    │             │  :6333      │
  └───────────┘             └─────────────┘
```

## Prerequisites

- **Neo4j** running on `bolt://localhost:7687` (default password: `password`)
- **Qdrant** running on `http://localhost:6333`
- **Python venv** with all packages installed (`pip install -r requirements.txt`)
- **Node.js** ≥ 18 (for frontend dev server)

## Step 1 — Configure `.env`

Edit `.env` in the project root:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

QDRANT_URL=http://localhost:6333

# Choose an LLM provider (auto-detected in this order: Gemini → Groq → OpenAI):
GEMINI_API_KEY=your_gemini_api_key   # Generous free tier — best for large PDFs
GROQ_API_KEY=your_groq_api_key       # gpt-oss-120b — fast, but tight free-tier limits
OPENAI_API_KEY=your_openai_api_key   # GPT-4o
# or run a local open-source model with no rate limits at all:
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1:8b
```

All LLM calls go through `llm/client.py` — see **Rate limits** below for tuning.

## Step 2 — Start Backend

```bash
# From the project root, with your venv activated
source ~/Desktop/RAG_VENV/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify backend is running:** http://localhost:8000/health  
**Interactive API docs:** http://localhost:8000/docs

## Step 3 — Start Frontend

```bash
cd frontend
npm run dev
```

Frontend opens at: **http://localhost:5173**

## Step 4 — Ingest a PDF

1. Open the frontend → **Upload PDF** tab
2. Drag and drop a medical PDF
3. Optionally enable **Link UMLS/MeSH/RxNorm IDs**
4. For long PDFs, enable **Large PDF mode** (1024-token chunks → half the LLM calls)
5. Click **Ingest PDF**

## Step 5 — Ask Questions

1. Go to **Ask Question** tab
2. Type a medical question (e.g., *"What drugs treat Type 2 Diabetes and interact with ACE inhibitors?"*)
3. Optionally enable **Dynamic Cypher** or **GDS Algorithms** for deeper retrieval
4. Click **Ask**

---

## Query rewriting

Before retrieval, `pipeline/query_rewriter.py` asks the LLM to clean up the question
(fix typos, expand abbreviations like *T2DM* or *BP*) and to split multi-part questions
into sub-queries, each searched separately. The answer still responds to the original
wording. It is on by default; turn it off with the **Query rewriting** checkbox or
`"use_query_rewrite": false` in the `/query` body. The response shows what was searched
(`rewritten_query`, `sub_queries`).

## Evaluation (RAGAS)

```bash
python evaluation/run_ragas.py      # answers testset.json with and without rewriting, then scores them
python evaluation/make_report.py    # builds evaluation/RAGAS_Evaluation_Report.pdf
```

Both steps cache per question in `evaluation/results/`, so a run stopped by the daily
quota resumes where it left off. Judge models are set in `METRIC_JUDGES`.

---

## Quick Start (both servers in parallel)

```bash
# Terminal 1 (project root, venv active)
uvicorn backend.main:app --port 8000 --reload

# Terminal 2
cd frontend && npm run dev
```

---

## Rate limits (long PDFs on free-tier APIs)

Ingestion makes **one LLM call per chunk**, so a 100-page PDF is ~200 calls of
~2,000 tokens each. Free tiers limit tokens per *minute* and per *day*.

What the code does for you:

| Mechanism | Where | Effect |
|---|---|---|
| Client-side throttle | `llm/rate_limiter.py` | Paces calls to stay under requests/min and tokens/min, so 429s are rare |
| Retry-after on 429 | `llm/client.py` | Waits exactly as long as the server asks, then retries |
| Daily-quota detection | `llm/client.py` | Stops cleanly (HTTP 429 / exit code 2) instead of hanging for hours |
| Per-chunk cache | `.cache/extractions/` | Re-uploading the same PDF later **resumes** — finished chunks cost nothing |
| All-or-nothing writes | `/ingest`, `run_stage1.py` | Nothing reaches Neo4j/Qdrant until every chunk is extracted |

What you can tune in `.env`:

```env
GROQ_MODEL=openai/gpt-oss-120b   # default; see "Choosing a Groq model" below
GROQ_TPM=8000             # tokens/min  (default 6000 — check console.groq.com/settings/limits)
GROQ_RPM=30               # requests/min

# Optionally use a different model just for bulk extraction
# (answers still use GROQ_MODEL)
# EXTRACTION_MODEL=openai/gpt-oss-20b

EXTRACTION_MAX_TOKENS=2048   # output budget per chunk (counts toward TPM)
LLM_MAX_WAIT_SECONDS=120     # longest 429 wait before giving up
```

The same `<PROVIDER>_RPM`, `<PROVIDER>_TPM` and `<PROVIDER>_MODEL` variables work
for `GEMINI`, `OPENAI` and `OLLAMA`.

### Choosing a Groq model

Benchmarked on this project's extraction prompt (free tier, Oct 2026):

| Model | Verdict |
|---|---|
| `openai/gpt-oss-120b` | **Default.** Most precise relations, fewest output tokens per chunk (more chunks/min) |
| `openai/gpt-oss-20b` | Slightly faster and finds more relations, but mislabels some (e.g. `TARGETS` vs `ACTIVATES`) |
| `qwen/qwen3.8-27b` | Accurate, but capped at 1,000 *output* tokens/min — far too slow for ingestion |

gpt-oss and qwen3 are reasoning models; `llm/client.py` keeps their hidden
"thinking" minimal (`LLM_REASONING_EFFORT`), because at `medium` the thinking
consumed the whole output budget and broke the JSON.

**Unlimited option:** install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`,
and set `LLM_PROVIDER=ollama`. It's slower per call but never rate-limited.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` | Activate your venv: `source ~/RAG_VENV/bin/activate` |
| `Connection refused :7687` | Start Neo4j: `neo4j start` or via Neo4j Desktop |
| `Connection refused :6333` | Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant` |
| Graph Explorer shows nothing | Upload a PDF first, then refresh |
| `CORS error` in browser | Use the Vite dev proxy — access frontend at `:5173`, not `:8000` directly |
| `Rate limit error (429)` on large PDFs | Daily quota used up — retry later (progress is cached), enable **Large PDF mode**, or see **Rate limits** above |
| `LLM provider error (401)` | The API key in `.env` is invalid or revoked — create a new one |
