import { useState, useEffect, useRef, useLayoutEffect } from 'react'
import { Icon } from './Icon'

// API base — relative so Vite proxy (dev) and FastAPI static (prod) both work
const API = ''

// Questions the sample PDFs in data/pdfs can answer (from evaluation/testset.json)
const EXAMPLES = [
  { kind: 'Drug interactions', icon: 'pill',
    q: 'Which diabetes drugs interact with blood-pressure medications?' },
  { kind: 'Shorthand question', icon: 'heart',
    q: 'pt on insulin + metoprolol, why check sugar more often?' },
  { kind: 'Mechanism, two hops', icon: 'flask',
    q: 'Which drug raises warfarin levels by inhibiting the enzyme that metabolizes it?' },
  { kind: 'Adverse effects', icon: 'kidney',
    q: 'What can cause acute kidney injury in a patient taking lisinopril and hydrochlorothiazide?' },
]

// Planner routes (pipeline/query_planner.py)
const ROUTE_TEXT = {
  hybrid_single: 'Graph + text, one hop',
  hybrid_multi:  'Graph + text, multi-hop',
  vector_only:   'Text search only',
  graph_gds:     'Graph algorithms',
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
    const label = part.slice(1, -1).trim().replace(/\.pdf/i, '')
    return (
      <span key={i} className="cite" title={part.slice(1, -1).trim()}>
        <Icon name="file" size={12} strokeWidth={2} />{label}
      </span>
    )
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
        <Icon name="check" size={15} strokeWidth={2.2} />
        <span>Completeness check passed: {parts > 1
          ? `all ${parts} parts of the question are answered.`
          : 'the question is fully answered.'}</span>
      </div>
    )
  }
  return (
    <div className="completeness repaired">
      <Icon name="refresh" size={15} strokeWidth={2} />
      <span>
        The first draft missed <strong>{info.missing.join('; ')}</strong>, so the answer was rewritten
        {info.added_sources?.length > 0 && <> with extra sources ({info.added_sources.join(', ')})</>}.
      </span>
    </div>
  )
}

const fileName = s => String(s || '').split(/[\\/]/).pop()

function Source({ item, index }) {
  const score = item.score ?? item.confidence ?? 0
  const isDoc = item.type === 'document'
  const pages = item.page != null ? [item.page] : (item.pages || [])
  return (
    <div className="source">
      <span className="source-idx">{index}</span>
      <div className="source-body">
        {isDoc ? (
          <div className="source-title" title={item.source}>
            <span className="name">{fileName(item.source) || 'Document'}</span>
          </div>
        ) : (
          <div className="source-path" title={item.path}>{item.path || 'Graph fact'}</div>
        )}
        {item.text && <div className="source-text">{item.text}</div>}
        <div className="source-meta">
          <span className="tag">{isDoc ? 'Text' : item.type === 'graph_gds' ? 'Algorithm' : 'Graph'}</span>
          {pages.length > 0 && <span>p. {pages.join(', ')}</span>}
          <span className="meter" title={`score ${score.toFixed(3)}`}>
            <span style={{ width: `${Math.max(4, Math.min(100, Math.round(score * 100)))}%` }} />
          </span>
          <span className="num">{score.toFixed(2)}</span>
        </div>
      </div>
    </div>
  )
}

function Switch({ checked, onChange, label, hint }) {
  return (
    <label className="switch">
      <input type="checkbox" role="switch" checked={checked} onChange={e => onChange(e.target.checked)} />
      <span className="switch-text">{label}<span className="switch-hint">{hint}</span></span>
    </label>
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

function Details({ result }) {
  const tabs = [
    { id: 'search', label: 'Retrieval', icon: 'route' },
    result.graph_paths?.length      && { id: 'paths',  label: 'Graph paths', icon: 'graph',  count: result.graph_paths.length },
    result.gds_paths?.length        && { id: 'gds',    label: 'Algorithms',  icon: 'layers', count: result.gds_paths.length },
    result.generated_cypher?.length && { id: 'cypher', label: 'Cypher',      icon: 'code',   count: result.generated_cypher.length },
  ].filter(Boolean)
  const [tab, setTab] = useState('search')
  const active = tabs.some(t => t.id === tab) ? tab : 'search'

  return (
    <div className="card">
      <div className="tabs" role="tablist" aria-label="How the answer was found">
        {tabs.map(t => (
          <button key={t.id} type="button" role="tab" className="tab"
            aria-selected={active === t.id} onClick={() => setTab(t.id)}>
            <Icon name={t.icon} size={15} />{t.label}
            {t.count != null && <span className="tab-count">{t.count}</span>}
          </button>
        ))}
      </div>
      <div className="tab-panel" role="tabpanel">
        {active === 'search' && (
          <dl className="kv">
            <dt>Route</dt>
            <dd>{ROUTE_TEXT[result.route] || result.route}{result.route_reason && <span className="muted"> · {result.route_reason}</span>}</dd>
            {result.retrieval_strategy && <><dt>Strategy</dt><dd>{result.retrieval_strategy}</dd></>}
            {result.rewritten_query && result.rewritten_query !== result.question && (
              <><dt>Searched as</dt><dd>{result.rewritten_query}</dd></>
            )}
            {result.sub_queries?.length > 0 && (
              <><dt>Sub-questions</dt>
                <dd><ol style={{ paddingLeft: 18 }}>{result.sub_queries.map((q, i) => <li key={i}>{q}</li>)}</ol></dd></>
            )}
            {result.vector_sources?.length > 0 && (
              <><dt>Passages read</dt>
                <dd className="chip-row">
                  {[...new Set(result.vector_sources)].map(s => {
                    const m = s.match(/^(.*?)\s+p\.(\d+)$/)
                    return <span key={s} className="chip">{fileName(m ? m[1] : s).replace(/\.pdf$/i, '')}{m && <span className="muted">p.{m[2]}</span>}</span>
                  })}
                </dd></>
            )}
            <dt>Planner confidence</dt><dd className="num">{Math.round((result.planner_confidence || 0) * 100)}%</dd>
          </dl>
        )}
        {active === 'paths' && (
          <div className="path-list">{result.graph_paths.map((p, i) => <div key={i} className="path-line">{p}</div>)}</div>
        )}
        {active === 'gds' && (
          <div className="path-list">{result.gds_paths.map((p, i) => <div key={i} className="path-line">{p}</div>)}</div>
        )}
        {active === 'cypher' && result.generated_cypher.map((q, i) => <pre key={i} className="code-block">{q}</pre>)}
      </div>
    </div>
  )
}

export function QueryPanel({ onQueryResult, onShowInGraph, backendDown = false }) {
  const [question, setQuestion]     = useState('')
  const [asked, setAsked]           = useState('')
  const [loading, setLoading]       = useState(false)
  const [result, setResult]         = useState(null)
  const [error, setError]           = useState(null)
  const [copied, setCopied]         = useState(false)
  const [showOptions, setShowOptions] = useState(false)
  const [allEntities, setAllEntities] = useState(false)
  const [useLLMCypher, setLLMCypher] = useState(false)
  const [useGDS, setUseGDS]         = useState(false)
  const [useRewrite, setUseRewrite] = useState(true)
  const [useCompleteness, setUseCompleteness] = useState(true)
  const inputRef = useRef(null)
  const elapsed  = useElapsed(loading)

  // Grow the box with the question (up to 240px); refit when its width changes
  useLayoutEffect(() => {
    const el = inputRef.current
    if (!el) return
    const fit = () => {
      if (!el.clientWidth) return   // hidden page: measure when shown
      el.style.height = 'auto'
      el.style.height = `${Math.min(el.scrollHeight, 240)}px`
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    return () => ro.disconnect()
  }, [question])

  const handleQuery = async (text = question) => {
    if (!text.trim() || loading) return
    setLoading(true)
    setAsked(text.trim())
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
      setError('Could not reach the API. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  const askExample = q => { setQuestion(q); handleQuery(q) }

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

  const reset = () => {
    setResult(null); setError(null); setQuestion(''); setAsked('')
    inputRef.current?.focus()
  }

  const started  = loading || result || error
  const optionsOn = [useRewrite, useCompleteness, useLLMCypher, useGDS].filter(Boolean).length
  const entities = [...new Set(result?.linked_entities || [])]

  return (
    <div className={`ask ${started ? 'has-result' : ''}`}>
      {!started && (
        <div className="ask-hero">
          <span className="ask-eyebrow"><Icon name="sparkle" size={14} />Knowledge graph + semantic search</span>
          <h1 className="ask-title">Ask a clinical question</h1>
          <p className="ask-sub">
            Answers combine facts from the Neo4j graph with passages from your PDFs,
            and cite the file and page they came from.
          </p>
        </div>
      )}
      {started && (
        <div className="page-header" style={{ marginBottom: 16 }}>
          <div>
            <h1 className="page-title">Ask</h1>
          </div>
          <div className="page-actions">
            <button type="button" className="btn btn-secondary" onClick={reset} disabled={loading}>
              <Icon name="plus" size={16} />New question
            </button>
          </div>
        </div>
      )}

      <div style={{ maxWidth: started ? 'none' : 760, margin: '0 auto' }}>
        <div className="composer">
          <label htmlFor="question" className="sr-only">Medical question</label>
          <textarea
            id="question"
            ref={inputRef}
            rows={1}
            placeholder="e.g. Which diabetes drugs interact with blood-pressure medications?"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKey}
          />
          <div className="composer-bar">
            <button type="button" className="options-toggle" aria-expanded={showOptions}
              aria-controls="ask-options" onClick={() => setShowOptions(o => !o)}>
              <Icon name="filter" size={14} />Options
              <span className="muted num">{optionsOn}/4</span>
              <Icon name="chevron" size={14} className="chev" />
            </button>
            <span className="composer-hint">Enter to ask · Shift+Enter for a new line</span>
            <button type="button" className="send-btn" onClick={() => handleQuery()}
              disabled={!question.trim() || loading} aria-label={loading ? 'Answering' : 'Ask'}>
              {loading
                ? <><span className="spinner" style={{ width: 14, height: 14 }} /><span className="num">{elapsed}s</span></>
                : <Icon name="send" size={18} strokeWidth={2.2} />}
            </button>
          </div>
          {showOptions && (
            <div className="options-panel" id="ask-options">
              <Switch checked={useRewrite} onChange={setUseRewrite} label="Query rewriting"
                hint="Fix typos, expand abbreviations, split multi-part questions" />
              <Switch checked={useCompleteness} onChange={setUseCompleteness} label="Completeness check"
                hint="Re-answer once if a part was missed (one extra LLM call)" />
              <Switch checked={useLLMCypher} onChange={setLLMCypher} label="LLM-written Cypher"
                hint="Let the model write a read-only graph query" />
              <Switch checked={useGDS} onChange={setUseGDS} label="Graph algorithms"
                hint="PageRank and shortest paths; needs Neo4j GDS" />
            </div>
          )}
        </div>

        {!started && (
          <>
            <div className="examples">
              <div className="section-label">Try one of these</div>
              <div className="examples-grid">
                {EXAMPLES.map(ex => (
                  <button key={ex.q} type="button" className="example"
                    onClick={() => askExample(ex.q)} disabled={backendDown}>
                    <span className="example-icon"><Icon name={ex.icon} size={17} /></span>
                    <span>
                      <span className="example-kind" style={{ display: 'block' }}>{ex.kind}</span>
                      <span className="example-q">{ex.q}</span>
                    </span>
                  </button>
                ))}
              </div>
            </div>
            <div className="ask-capabilities">
              <span><Icon name="database" size={15} />Qdrant semantic search</span>
              <span><Icon name="graph" size={15} />Neo4j knowledge graph</span>
              <span><Icon name="file" size={15} />Answers cite file and page</span>
            </div>
          </>
        )}
      </div>

      {error && (
        <div className="alert error" role="alert" style={{ marginTop: 20 }}>
          <Icon name="alert" size={17} />
          <div className="alert-body"><strong>No answer.</strong> {error}</div>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => handleQuery(asked)}>
            <Icon name="refresh" size={14} />Retry
          </button>
        </div>
      )}

      {loading && (
        <div className="result-grid">
          <div className="result-main">
            <div className="card">
              <div className="working" role="status">
                <span className="spinner" />
                <span>Working on it… <span className="num muted">{elapsed}s</span></span>
                <span className="working-steps">
                  <span>Rewrite</span><span>Search text</span><span>Walk graph</span><span>Write answer</span>
                </span>
              </div>
              <div className="answer-skeleton" aria-hidden="true">
                <div className="skeleton" style={{ width: '92%' }} />
                <div className="skeleton" style={{ width: '86%' }} />
                <div className="skeleton" style={{ width: '64%' }} />
                <div className="skeleton" style={{ width: '88%', marginTop: 10 }} />
                <div className="skeleton" style={{ width: '72%' }} />
              </div>
            </div>
            <p className="muted" style={{ fontSize: 13 }}>
              Free-tier rate limits can make this take a minute or two.
            </p>
          </div>
          <div className="result-rail">
            <div className="card card-pad answer-skeleton" aria-hidden="true">
              <div className="skeleton" style={{ width: '40%' }} />
              <div className="skeleton" style={{ width: '90%' }} />
              <div className="skeleton" style={{ width: '75%' }} />
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="result-grid">
          <div className="result-main">
            <article className="card answer-card">
              <div className="card-head">
                <span className="card-title">Answer</span>
                <div className="answer-meta">
                  <span className="tag accent">{ROUTE_TEXT[result.route] || result.route}</span>
                  <span className="tag" title="Time to answer"><Icon name="clock" size={13} />{(result.latency_ms / 1000).toFixed(1)}s</span>
                  <button type="button" className="btn btn-ghost btn-sm" onClick={copyAnswer}>
                    <Icon name={copied ? 'check' : 'copy'} size={14} />{copied ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </div>
              <div className="answer-body"><AnswerText text={result.answer} /></div>
              {result.completeness && (
                <div className="answer-foot"><CompletenessNote info={result.completeness} /></div>
              )}
            </article>
            <Details result={result} />
          </div>

          <aside className="result-rail">
            {result.provenance?.length > 0 && (
              <div className="card">
                <div className="card-head">
                  <span className="card-title">Sources</span>
                  <span className="card-sub num" style={{ marginLeft: 'auto' }}>{result.provenance.length}</span>
                </div>
                <div className="source-list">
                  {result.provenance.map((p, i) => <Source key={i} item={p} index={i + 1} />)}
                </div>
              </div>
            )}
            {entities.length > 0 && (
              <div className="card entity-card">
                <div className="card-head">
                  <span className="card-title">Graph entities</span>
                  <span className="card-sub num" style={{ marginLeft: 'auto' }}>{entities.length}</span>
                </div>
                <div className="card-pad">
                  <div className="chip-row">
                    {(allEntities ? entities : entities.slice(0, 12)).map(e => <span key={e} className="chip">{e}</span>)}
                    {entities.length > 12 && (
                      <button type="button" className="chip" style={{ cursor: 'pointer' }}
                        onClick={() => setAllEntities(a => !a)}>
                        {allEntities ? 'Show fewer' : `+${entities.length - 12} more`}
                      </button>
                    )}
                  </div>
                  {onShowInGraph && (
                    <button type="button" className="btn btn-secondary" onClick={onShowInGraph}>
                      <Icon name="graph" size={16} />Show in graph
                    </button>
                  )}
                </div>
              </div>
            )}
          </aside>
        </div>
      )}
    </div>
  )
}
