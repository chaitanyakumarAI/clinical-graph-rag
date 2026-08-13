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
- **Python venv** with all packages installed (`RAG_VENV` or similar)
- **Node.js** ≥ 18 (for frontend dev server)

## Step 1 — Configure `.env`

Edit `.env` in the project root:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

QDRANT_URL=http://localhost:6333

# Choose one LLM provider:
GROQ_API_KEY=your_groq_api_key
# or:
OPENAI_API_KEY=your_openai_api_key
```

## Step 2 — Start Backend

```bash
# Option A: Use the start script (auto-detects RAG_VENV)
./start_backend.sh

# Option B: Explicit venv path
VENV_PATH=~/RAG_VENV ./start_backend.sh

# Option C: Manual (activate venv first, then from project root)
source ~/RAG_VENV/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify backend is running:** http://localhost:8000/health  
**Interactive API docs:** http://localhost:8000/docs

## Step 3 — Start Frontend

```bash
# Using the start script
./start_frontend.sh

# Or manually
cd frontend
npm run dev
```

Frontend opens at: **http://localhost:5173**

## Step 4 — Ingest a PDF

1. Open the frontend → **Upload PDF** tab
2. Drag and drop a medical PDF
3. Optionally enable **Link UMLS/MeSH/RxNorm IDs**
4. Click **Ingest PDF**

## Step 5 — Ask Questions

1. Go to **Ask Question** tab
2. Type a medical question (e.g., *"What drugs treat Type 2 Diabetes and interact with ACE inhibitors?"*)
3. Optionally enable **Dynamic Cypher** or **GDS Algorithms** for deeper retrieval
4. Click **Ask**

---

## Quick Start (both servers in parallel)

```bash
# Terminal 1
VENV_PATH=~/RAG_VENV ./start_backend.sh

# Terminal 2
./start_frontend.sh
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` | Activate your venv: `source ~/RAG_VENV/bin/activate` |
| `Connection refused :7687` | Start Neo4j: `neo4j start` or via Neo4j Desktop |
| `Connection refused :6333` | Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant` |
| Graph Explorer shows nothing | Upload a PDF first, then refresh |
| `CORS error` in browser | Use the Vite dev proxy — access frontend at `:5173`, not `:8000` directly |
