import { useState, useEffect, useRef } from 'react'

// API base — relative so Vite proxy (dev) and FastAPI static (prod) both work
const API = ''

// Questions the sample PDFs in data/pdfs can answer (from evaluation/testset.json)
const EXAMPLES = [
  'Which diabetes drugs interact with blood-pressure medications?',
  'pt on insulin + metoprolol, why check sugar more often?',
  'Which drug raises warfarin levels by inhibiting the enzyme that metabolizes it?',
  'What can cause acute kidney injury in a patient taking lisinopril and hydrochlorothiazide?',
]

function RoutePill({ route }) {
  return <span className={`route-pill ${route}`}>{route.replace('_', ' ')}</span>
}

// The LLM answers in light Markdown and cites as [file.pdf, p.2] or
// 【file.pdf, p.2】. Render that subset as React elements (no innerHTML, so
// nothing in an answer can inject markup).
const INLINE = /(\*\*[^*]+\*\*|\[[^[\]]*?\.pdf[^[\]]*\]|【[^【】]*?\.pdf[^【】]*】|`[^`]+`|\*[^*\s][^*]*\*)/gi

function Inline({ text }) {
  return text.split(INLINE).map((part, i) => {
    if (i % 2 === 0) return part
    if (part.startsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`'))  return <code key={i}>{part.slice(1, -1)}</code>
    if (part.startsWith('*'))  return <em key={i}>{part.slice(1, -1)}</em>
    return <span key={i} className="cite">{part.slice(1, -1).trim()}</span>
  })
}

function AnswerText({ text }) {
  const blocks = []
  let list = null
  text.split('\n').forEach(raw => {
    const line   = raw.trim()
    const bullet = line.match(/^(?:[-*•]|\d+[.)])\s+(.*)$/)
    if (bullet) {
      if (!list) { list = { ordered: /^\d/.test(line), items: [] }; blocks.push(list) }
      list.items.push(bullet[1])
      return
    }
    list = null
    if (!line) return
    const heading = line.match(/^#{1,6}\s+(.*)$/)
    blocks.push(heading ? { heading: heading[1] } : { para: line })
  })
  return blocks.map((b, i) => {
    if (b.items) {
      const List = b.ordered ? 'ol' : 'ul'
      return <List key={i}>{b.items.map((item, j) => <li key={j}><Inline text={item} /></li>)}</List>
    }
    if (b.heading) return <p key={i} className="answer-heading"><Inline text={b.heading} /></p>
    return <p key={i}><Inline text={b.para} /></p>
  })
}

function CompletenessNote({ info }) {
  if (!info) return null
  const parts = info.parts?.length || 0
  if (!info.repaired) {
    return (
      <div className="completeness ok">
        ✓ Completeness check: all {parts} part{parts !== 1 ? 's' : ''} of the question answered
      </div>
    )
  }
  return (
    <div className="completeness repaired">
      ↻ Completeness check: the first draft missed{' '}
      <strong>{info.missing.join('; ')}</strong>, so the answer was regenerated
      {info.added_sources?.length > 0 && <> with {info.added_sources.join(', ')}</>}.
    </div>
  )
}

// ── Citation card — shows actual filename + page number ────────────────────

function ProvenanceCard({ item }) {
  const conf = item.confidence || item.score || 0

  // source may be: "path/to/file.pdf", "file.pdf", or an internal ID
  const rawSource  = item.source || ''
  const filename   = rawSource.split('/').pop() || rawSource
  const pageNum    = item.page != null ? item.page : null
  const isDocument = item.type === 'document'
  const isGraph    = item.type === 'graph' || item.type === 'graph_gds'

  return (
    <div className="prov-card">
      <div className={`prov-card-type ${item.type}`}>{item.type.replace('_', ' ')}</div>

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
        {isGraph && item.pages?.length > 0 && (
          <span style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 4 }}>
            {item.pages.map(p => (
              <span key={p} className="prov-page-badge">p.{p}</span>
            ))}
          </span>
        )}
      </div>

      {item.text && (
        <div className="prov-card-text">
          {item.text.slice(0, 130)}{item.text.length > 130 ? '…' : ''}
        </div>
      )}

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
      <button type="button" className="collapsible-header"
        aria-expanded={open} onClick={() => setOpen(!open)}>
        <span>{title}</span>
        <span aria-hidden="true">{open ? '▲' : '▼'}</span>
      </button>
      {open && <div style={{ paddingTop: 8 }}>{children}</div>}
    </div>
  )
}

// Ticks once a second while a request runs; answers take several seconds
function useElapsed(running) {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    if (!running) return
    setSeconds(0)
    const start = Date.now()
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [running])
  return seconds
}

export function QueryPanel({ onQueryResult, onShowInGraph, backendDown = false }) {
  const [question, setQuestion]     = useState('')
  const [loading, setLoading]       = useState(false)
  const [result, setResult]         = useState(null)
  const [error, setError]           = useState(null)
  const [copied, setCopied]         = useState(false)
  const [useLLMCypher, setLLMCypher] = useState(false)
  const [useGDS, setUseGDS]         = useState(false)
  const [useRewrite, setUseRewrite] = useState(true)
  const [useCompleteness, setUseCompleteness] = useState(true)
  const inputRef = useRef(null)
  const elapsed  = useElapsed(loading)

  const handleQuery = async (text = question) => {
    if (!text.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    setCopied(false)
    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          provider: 'auto',
          top_k: 5, top_n: 8,
          use_query_rewrite: useRewrite,
          use_llm_cypher: useLLMCypher,
          use_gds: useGDS,
          use_completeness_check: useCompleteness,
        }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        // 429 = LLM rate limit used up, 502 = bad API key / model, 503 = a store is down
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

  const askExample = q => {
    setQuestion(q)
    handleQuery(q)
  }

  const handleKey = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleQuery() }
  }

  const copyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(result.answer)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard blocked: nothing to do */ }
  }

  return (
    <div className="card">
      <div className="card-title">🔬 Ask a Medical Question</div>

      <div className="query-input-row">
        <textarea
          ref={inputRef}
          className="query-input"
          rows={2}
          aria-label="Medical question"
          placeholder="Which drugs for Type 2 Diabetes interact with hypertension medications?"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={handleKey}
        />
        <button className="btn btn-primary" onClick={() => handleQuery()}
          disabled={!question.trim() || loading}
          style={{ alignSelf: 'flex-start', minWidth: 100 }}>
          {loading ? <><span className="spinner" /> {elapsed}s</> : '🚀 Ask'}
        </button>
      </div>
      <div className="input-hint">Enter to ask · Shift+Enter for a new line</div>

      <div className="query-options">
        <label className="toggle-row" title="Fix typos, expand abbreviations and split multi-part questions before searching">
          <input type="checkbox" checked={useRewrite} onChange={e => setUseRewrite(e.target.checked)} />
          Query rewriting
        </label>
        <label className="toggle-row" title="For multi-part questions: check every part was answered and regenerate once if not (one extra LLM call)">
          <input type="checkbox" checked={useCompleteness} onChange={e => setUseCompleteness(e.target.checked)} />
          Completeness check
        </label>
        <label className="toggle-row" title="Let the LLM write a read-only Cypher query for the graph">
          <input type="checkbox" checked={useLLMCypher} onChange={e => setLLMCypher(e.target.checked)} />
          LLM-written Cypher
        </label>
        <label className="toggle-row" title="PageRank / shortest-path context; needs the Neo4j Graph Data Science plugin">
          <input type="checkbox" checked={useGDS} onChange={e => setUseGDS(e.target.checked)} />
          Graph algorithms (GDS)
        </label>
      </div>

      {!result && !loading && (
        <div className="examples">
          <div className="examples-label">Try an example</div>
          <div className="examples-row">
            {EXAMPLES.map(q => (
              <button key={q} type="button" className="example-chip"
                onClick={() => askExample(q)} disabled={backendDown}>
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading && (
        <>
          <div className="progress-bar"><div className="progress-bar-fill" /></div>
          <div className="input-hint" role="status">
            Rewriting, searching Qdrant and Neo4j, then writing a cited answer…
          </div>
        </>
      )}

      {error && <div className="alert error" role="alert">⚠️ {error}</div>}

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
            <span className="badge" style={{ marginLeft: 'auto' }}
              title="How sure the planner was about the route">
              planner {Math.round((result.planner_confidence || 0) * 100)}%
            </span>
            <span className="badge">{(result.latency_ms / 1000).toFixed(1)}s</span>
          </div>

          {/* What retrieval searched for, when it differs from the typed question */}
          {(result.rewritten_query !== result.question || result.sub_queries?.length > 0) && (
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              🔁 Searched as: <em>{result.rewritten_query}</em>
              {result.sub_queries?.length > 0 && (
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
                  {result.sub_queries.map((q, i) => <span key={i} className="chip">{q}</span>)}
                </div>
              )}
            </div>
          )}

          {/* Answer */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
              <div className="card-title" style={{ margin: 0 }}>Answer</div>
              <button type="button" className="btn btn-secondary btn-sm"
                style={{ marginLeft: 'auto' }} onClick={copyAnswer}>
                {copied ? '✓ Copied' : '⧉ Copy'}
              </button>
            </div>
            <div className="answer-box"><AnswerText text={result.answer} /></div>
            <CompletenessNote info={result.completeness} />
          </div>

          {/* Linked entities */}
          {result.linked_entities?.length > 0 && (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                <div className="card-title" style={{ margin: 0 }}>Linked Graph Entities</div>
                {onShowInGraph && (
                  <button type="button" className="btn btn-secondary btn-sm"
                    style={{ marginLeft: 'auto' }} onClick={onShowInGraph}>
                    🕸️ Show in graph
                  </button>
                )}
              </div>
              <div className="chip-row">
                {[...new Set(result.linked_entities)].map(e => <span key={e} className="chip">{e}</span>)}
              </div>
            </div>
          )}

          {result.provenance?.length > 0 && (
            <Collapsible title={`📎 Provenance & Citations (${result.provenance.length})`} defaultOpen>
              <div className="provenance-grid">
                {result.provenance.map((p, i) => <ProvenanceCard key={i} item={p} />)}
              </div>
            </Collapsible>
          )}

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

          {result.graph_paths?.length > 0 && (
            <Collapsible title={`🕸️ Graph Paths (${result.graph_paths.length})`}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {result.graph_paths.map((p, i) => (
                  <div key={i} className="path-line teal">{p}</div>
                ))}
              </div>
            </Collapsible>
          )}

          {result.gds_paths?.length > 0 && (
            <Collapsible title={`🧮 GDS Algorithm Context (${result.gds_paths.length})`}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {result.gds_paths.map((p, i) => (
                  <div key={i} className="path-line purple">{p}</div>
                ))}
              </div>
            </Collapsible>
          )}

          {result.generated_cypher?.length > 0 && (
            <Collapsible title={`⚡ Generated Cypher (${result.generated_cypher.length} queries)`}>
              {result.generated_cypher.map((q, i) => (
                <div key={i} className="cypher-block">{q}</div>
              ))}
            </Collapsible>
          )}

          <button type="button" className="btn btn-secondary btn-sm" style={{ alignSelf: 'flex-start' }}
            onClick={() => { setResult(null); setQuestion(''); inputRef.current?.focus() }}>
            ＋ New question
          </button>
        </div>
      )}
    </div>
  )
}
