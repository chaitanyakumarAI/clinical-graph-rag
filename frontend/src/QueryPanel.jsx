import { useState } from 'react'

// API base — relative so Vite proxy (dev) and FastAPI static (prod) both work
const API = ''

function RoutePill({ route }) {
  return <span className={`route-pill ${route}`}>{route.replace('_', ' ')}</span>
}

// ── Citation card — shows actual filename + page number ────────────────────

function ProvenanceCard({ item }) {
  const conf = item.confidence || item.score || 0

  // Resolve display-friendly filename from source string
  // source may be: "path/to/file.pdf", "file.pdf", or an internal ID
  const rawSource  = item.source || ''
  const filename   = rawSource.split('/').pop() || rawSource   // last path segment
  const pageNum    = item.page != null ? item.page : null
  const isDocument = item.type === 'document'
  const isGraph    = item.type === 'graph' || item.type === 'graph_gds'

  return (
    <div className="prov-card">
      {/* Type badge */}
      <div className={`prov-card-type ${item.type}`}>{item.type}</div>

      {/* Citation line */}
      <div className="prov-card-citation">
        {isDocument && (
          <>
            <span className="prov-filename" title={rawSource}>
              📄 {filename || '—'}
            </span>
            {pageNum != null && (
              <span className="prov-page-badge">p.{pageNum}</span>
            )}
          </>
        )}
        {isGraph && item.path && (
          <span className="prov-graph-path" title={item.path}>
            🕸 {item.path.slice(0, 70)}{item.path.length > 70 ? '…' : ''}
          </span>
        )}
        {/* Graph edge page provenance */}
        {isGraph && item.pages?.length > 0 && (
          <span style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
            {item.pages.map(p => (
              <span key={p} className="prov-page-badge">p.{p}</span>
            ))}
          </span>
        )}
      </div>

      {/* Text snippet (document only) */}
      {item.text && (
        <div className="prov-card-text">
          {item.text.slice(0, 130)}{item.text.length > 130 ? '…' : ''}
        </div>
      )}

      {/* Confidence bar */}
      <div className="conf-bar">
        <div className="conf-bar-fill" style={{ width: `${Math.round(conf * 100)}%` }} />
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 4 }}>
        score {(item.score || conf).toFixed(3)}
      </div>
    </div>
  )
}

function Collapsible({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <div className="collapsible-header" onClick={() => setOpen(!open)}>
        <span>{title}</span>
        <span>{open ? '▲' : '▼'}</span>
      </div>
      {open && <div style={{ paddingTop: 8 }}>{children}</div>}
    </div>
  )
}

export function QueryPanel({ onQueryResult }) {
  const [question, setQuestion]     = useState('')
  const [loading, setLoading]       = useState(false)
  const [result, setResult]         = useState(null)
  const [error, setError]           = useState(null)
  const [useLLMCypher, setLLMCypher] = useState(false)
  const [useGDS, setUseGDS]         = useState(false)

  const handleQuery = async () => {
    if (!question.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          provider: 'auto',
          top_k: 5, top_n: 8,
          use_llm_cypher: useLLMCypher,
          use_gds: useGDS,
        }),
      })
      const data = await res.json()
      if (!res.ok) {
        // e.g. 429 = LLM rate limit used up, 502 = bad API key / model
        setError(data.detail || `Request failed (HTTP ${res.status})`)
        return
      }
      setResult(data)
      onQueryResult?.(data)
    } catch {
      setError('Failed to connect to API. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  const handleKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleQuery() }
  }

  return (
    <div className="card">
      <div className="card-title">🔬 Ask a Medical Question</div>

      <div className="query-input-row">
        <textarea
          className="query-input"
          rows={2}
          placeholder="Which drugs for Type 2 Diabetes interact with hypertension medications?"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={handleKey}
        />
        <button className="btn btn-primary" onClick={handleQuery}
          disabled={!question.trim() || loading}
          style={{ alignSelf: 'flex-start', minWidth: 100 }}>
          {loading ? <><span className="spinner" /> Running…</> : '🚀 Ask'}
        </button>
      </div>

      <div className="query-options">
        <label className="toggle-row">
          <input type="checkbox" checked={useLLMCypher} onChange={e => setLLMCypher(e.target.checked)} />
          Dynamic Cypher (Phase 3)
        </label>
        <label className="toggle-row">
          <input type="checkbox" checked={useGDS} onChange={e => setUseGDS(e.target.checked)} />
          GDS Algorithms (Phase 4)
        </label>
        <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
          Qdrant-first pipeline always active (Phase 5)
        </span>
      </div>

      {loading && <div className="progress-bar"><div className="progress-bar-fill" /></div>}

      {error && (
        <div style={{
          marginTop: 12, padding: '10px 14px',
          background: 'rgba(248,81,73,0.1)',
          border: '1px solid rgba(248,81,73,0.3)',
          borderRadius: 8, fontSize: 13, color: 'var(--accent-red)',
        }}>
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Route + strategy + latency */}
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <RoutePill route={result.route} />
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {result.route_reason}
            </span>
            {result.retrieval_strategy && (
              <span style={{ fontSize: 11, color: 'var(--text-dim)', fontStyle: 'italic' }}>
                — {result.retrieval_strategy}
              </span>
            )}
            <span className="badge" style={{ marginLeft: 'auto' }}>
              planner {Math.round((result.planner_confidence || 0) * 100)}%
            </span>
            <span className="badge">{result.latency_ms}ms</span>
          </div>

          {/* Answer */}
          <div>
            <div className="card-title" style={{ marginBottom: 8 }}>Answer</div>
            <div className="answer-box">{result.answer}</div>
          </div>

          {/* Linked entities */}
          {result.linked_entities?.length > 0 && (
            <div>
              <div className="card-title" style={{ marginBottom: 8 }}>Linked Graph Entities</div>
              <div className="chip-row">
                {result.linked_entities.map(e => <span key={e} className="chip">{e}</span>)}
              </div>
            </div>
          )}

          {/* Provenance — now shows filename + page */}
          {result.provenance?.length > 0 && (
            <Collapsible title={`📎 Provenance & Citations (${result.provenance.length})`} defaultOpen>
              <div className="provenance-grid">
                {result.provenance.map((p, i) => <ProvenanceCard key={i} item={p} />)}
              </div>
            </Collapsible>
          )}

          {/* Vector sources — filename + page chips */}
          {result.vector_sources?.length > 0 && (
            <Collapsible title={`📚 Document Sources (${result.vector_sources.length})`}>
              <div className="chip-row">
                {result.vector_sources.map((s, i) => {
                  // s = "some/path/file.pdf p.3"  — parse it
                  const match = s.match(/^(.*?)\s+p\.(\d+)$/)
                  const fname = match
                    ? match[1].split('/').pop()
                    : s.split('/').pop()
                  const page  = match ? match[2] : null
                  return (
                    <span key={i} className="source-chip">
                      <span className="source-chip-name">📄 {fname}</span>
                      {page && <span className="source-chip-page">p.{page}</span>}
                    </span>
                  )
                })}
              </div>
            </Collapsible>
          )}

          {/* Graph paths */}
          {result.graph_paths?.length > 0 && (
            <Collapsible title={`🕸️ Graph Paths (${result.graph_paths.length})`}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {result.graph_paths.map((p, i) => (
                  <div key={i} style={{
                    fontSize: 12, fontFamily: 'monospace',
                    padding: '6px 10px', background: 'var(--bg-input)',
                    borderRadius: 6, color: 'var(--accent-teal)',
                  }}>
                    {p}
                  </div>
                ))}
              </div>
            </Collapsible>
          )}

          {/* GDS paths */}
          {result.gds_paths?.length > 0 && (
            <Collapsible title={`🧮 GDS Algorithm Context (${result.gds_paths.length})`}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {result.gds_paths.map((p, i) => (
                  <div key={i} style={{
                    fontSize: 12, fontFamily: 'monospace',
                    padding: '6px 10px', background: 'var(--bg-input)',
                    borderRadius: 6, color: 'var(--accent-purple)',
                  }}>
                    {p}
                  </div>
                ))}
              </div>
            </Collapsible>
          )}

          {/* Generated Cypher */}
          {result.generated_cypher?.length > 0 && (
            <Collapsible title={`⚡ Generated Cypher (${result.generated_cypher.length} queries)`}>
              {result.generated_cypher.map((q, i) => (
                <div key={i} className="cypher-block">{q}</div>
              ))}
            </Collapsible>
          )}
        </div>
      )}
    </div>
  )
}
