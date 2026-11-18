import { useState, useCallback, useEffect } from 'react'
import './index.css'
import { UploadPanel }    from './UploadPanel'
import { QueryPanel }     from './QueryPanel'
import { GraphExplorer }  from './GraphExplorer'
import { StatsDashboard } from './StatsDashboard'

const PAGES = [
  { id: 'query',  label: 'Ask Question',   icon: '🔬' },
  { id: 'upload', label: 'Upload PDF',     icon: '📄' },
  { id: 'graph',  label: 'Graph Explorer', icon: '🕸️' },
  { id: 'stats',  label: 'Dashboard',      icon: '📊' },
]

const HEALTH_POLL_MS = 30000

const pageFromHash = () => {
  const id = window.location.hash.slice(1)
  return PAGES.some(p => p.id === id) ? id : 'query'
}

// Polls /health so a stopped backend or database shows up in the header
// instead of as a confusing error after the next click.
function useBackendStatus() {
  const [status, setStatus] = useState({ state: 'checking' })
  useEffect(() => {
    let alive = true
    const check = async () => {
      try {
        const res  = await fetch('/health')
        const data = await res.json()
        if (alive) setStatus(data.status === 'ok'
          ? { state: 'ok', version: data.version }
          : { state: 'degraded', detail: data.error })
      } catch {
        if (alive) setStatus({ state: 'offline' })
      }
    }
    check()
    const timer = setInterval(check, HEALTH_POLL_MS)
    return () => { alive = false; clearInterval(timer) }
  }, [])
  return status
}

const STATUS_TEXT = {
  checking: 'Checking…',
  ok:       'Connected',
  degraded: 'Neo4j unreachable',
  offline:  'Backend offline',
}

export default function App() {
  const [page, setPageState]              = useState(pageFromHash)
  const [highlightEntities, setHighlight] = useState([])
  // Shared document list — fetched in UploadPanel, used by GraphExplorer filter
  const [documents, setDocuments]         = useState([])
  const status = useBackendStatus()

  const setPage = useCallback(id => {
    setPageState(id)
    window.scrollTo(0, 0)
    if (window.location.hash !== `#${id}`) window.history.pushState(null, '', `#${id}`)
  }, [])

  // Back / forward buttons move between pages
  useEffect(() => {
    const onPop = () => setPageState(pageFromHash())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const handleQueryResult = useCallback(result => {
    setHighlight([...new Set(result.linked_entities || [])])
  }, [])

  const handleDocumentsChange = useCallback(docs => {
    setDocuments(docs)
  }, [])

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="header">
        <div className="header-logo" aria-hidden="true">🧬</div>
        <span className="header-title">Medical Graph RAG</span>
        {status.version && <span className="header-subtitle">v{status.version}</span>}
        <div className="header-badge">
          <span className="badge hide-narrow">Neo4j</span>
          <span className="badge hide-narrow">Qdrant</span>
          <span className={`status-pill ${status.state}`} role="status"
            title={status.detail || STATUS_TEXT[status.state]}>
            <span className="status-dot" aria-hidden="true" />
            {STATUS_TEXT[status.state]}
          </span>
        </div>
      </header>

      {/* Sidebar (a tab bar on narrow screens) */}
      <nav className="sidebar" aria-label="Pages">
        {PAGES.map(p => (
          <button
            key={p.id}
            type="button"
            className={`nav-item ${page === p.id ? 'active' : ''}`}
            aria-current={page === p.id ? 'page' : undefined}
            onClick={() => setPage(p.id)}
          >
            <span className="nav-icon" aria-hidden="true">{p.icon}</span>
            <span className="nav-label">{p.label}</span>
          </button>
        ))}

        {documents.length > 0 && (
          <button type="button" className="nav-item nav-docs" onClick={() => setPage('upload')}>
            <span className="nav-icon" aria-hidden="true">📚</span>
            <span style={{ fontSize: 12 }}>
              {documents.length} doc{documents.length !== 1 ? 's' : ''} ingested
            </span>
          </button>
        )}

        <div className="sidebar-footer">
          <div>🔎 Qdrant semantic search</div>
          <div>🕸️ Neo4j knowledge graph</div>
          <div>📎 Answers cite file and page</div>
        </div>
      </nav>

      {/* Main. Query and upload stay mounted so an answer, or an upload in
          progress, survives switching pages. */}
      <main className="main-content">
        <div hidden={page !== 'query'}>
          <QueryPanel
            onQueryResult={handleQueryResult}
            onShowInGraph={() => setPage('graph')}
            backendDown={status.state === 'offline'}
          />
        </div>
        <div hidden={page !== 'upload'}>
          <UploadPanel
            onSuccess={() => {}}
            onDocumentsChange={handleDocumentsChange}
          />
        </div>
        {page === 'graph' && (
          <GraphExplorer
            highlightEntities={highlightEntities}
            documents={documents}
            onClearHighlight={() => setHighlight([])}
          />
        )}
        {page === 'stats' && <StatsDashboard />}
      </main>
    </div>
  )
}
