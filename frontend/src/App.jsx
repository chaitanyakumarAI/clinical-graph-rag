import { useState, useCallback, useEffect } from 'react'
import './index.css'
import { UploadPanel }    from './UploadPanel'
import { QueryPanel }     from './QueryPanel'
import { GraphExplorer }  from './GraphExplorer'
import { StatsDashboard } from './StatsDashboard'
import { Icon, Logo }     from './Icon'

const PAGES = [
  { id: 'query',  label: 'Ask',       icon: 'ask'    },
  { id: 'graph',  label: 'Graph',     icon: 'graph'  },
  { id: 'upload', label: 'Documents', icon: 'file'   },
  { id: 'stats',  label: 'Overview',  icon: 'chart'  },
]

const HEALTH_POLL_MS = 30000

const pageFromHash = () => {
  const id = window.location.hash.slice(1)
  return PAGES.some(p => p.id === id) ? id : 'query'
}

// Polls /health so a stopped backend or database shows in the sidebar
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
  checking: 'Checking connection…',
  ok:       'All systems connected',
  degraded: 'Neo4j unreachable',
  offline:  'Backend offline',
}

const readPref = () => {
  try { return localStorage.getItem('theme') } catch { return null }
}

// Light/dark: follows the OS until the user picks one with the toggle
const DARK_MQ = window.matchMedia('(prefers-color-scheme: dark)')

function useTheme() {
  const [pref, setPref]     = useState(readPref)
  const [osDark, setOsDark] = useState(DARK_MQ.matches)

  useEffect(() => {
    const onChange = e => setOsDark(e.matches)
    DARK_MQ.addEventListener('change', onChange)
    return () => DARK_MQ.removeEventListener('change', onChange)
  }, [])

  const theme = pref === 'light' || pref === 'dark' ? pref : (osDark ? 'dark' : 'light')

  useEffect(() => {
    if (pref) document.documentElement.dataset.theme = pref
    else delete document.documentElement.dataset.theme
  }, [pref])

  const toggle = useCallback(() => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setPref(next)
    try { localStorage.setItem('theme', next) } catch { /* private mode */ }
  }, [theme])

  return [theme, toggle]
}

export default function App() {
  const [page, setPageState]              = useState(pageFromHash)
  const [highlightEntities, setHighlight] = useState([])
  // Shared document list: fetched in UploadPanel, used by the graph filter
  const [documents, setDocuments]         = useState([])
  const status = useBackendStatus()
  const [theme, toggleTheme] = useTheme()

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

  const handleDocumentsChange = useCallback(docs => setDocuments(docs), [])

  const statusMeta = status.state === 'ok'
    ? `Neo4j · Qdrant · v${status.version}`
    : status.state === 'offline' ? 'Start the API on port 8000'
    : status.detail || 'Neo4j · Qdrant'

  const themeLabel = theme === 'dark' ? 'Light theme' : 'Dark theme'

  return (
    <div className="app">
      {/* Sidebar (wide screens) */}
      <aside className="sidebar">
        <div className="brand">
          <Logo size={32} />
          <div>
            <div className="brand-name">Clinical Graph RAG</div>
            <div className="brand-sub">Graph + vector answers</div>
          </div>
        </div>

        <nav className="nav" aria-label="Pages">
          {PAGES.map(p => (
            <button key={p.id} type="button"
              className={`nav-item ${page === p.id ? 'active' : ''}`}
              aria-current={page === p.id ? 'page' : undefined}
              onClick={() => setPage(p.id)}>
              <Icon name={p.icon} />
              {p.label}
              {p.id === 'upload' && documents.length > 0 && (
                <span className="nav-count" aria-label={`${documents.length} documents`}>{documents.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">
          <div className="status-card" role="status" title={status.detail || ''}>
            <div className="status-line">
              <span className={`status-dot ${status.state}`} aria-hidden="true" />
              {STATUS_TEXT[status.state]}
            </div>
            <div className="status-meta">{statusMeta}</div>
          </div>
          <button type="button" className="theme-toggle" onClick={toggleTheme}>
            <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={17} />
            {themeLabel}
          </button>
        </div>
      </aside>

      {/* Top bar + bottom tabs (narrow screens) */}
      <header className="topbar">
        <Logo size={28} />
        <span className="brand-name">Clinical Graph RAG</span>
        <div className="topbar-right">
          <span className={`status-dot ${status.state}`} role="status"
            aria-label={STATUS_TEXT[status.state]} title={STATUS_TEXT[status.state]} />
          <button type="button" className="btn btn-ghost btn-icon" onClick={toggleTheme}
            aria-label={themeLabel}>
            <Icon name={theme === 'dark' ? 'sun' : 'moon'} />
          </button>
        </div>
      </header>
      <nav className="tabbar" aria-label="Pages">
        {PAGES.map(p => (
          <button key={p.id} type="button"
            className={`tab-btn ${page === p.id ? 'active' : ''}`}
            aria-current={page === p.id ? 'page' : undefined}
            onClick={() => setPage(p.id)}>
            <Icon name={p.icon} size={20} />
            {p.label}
          </button>
        ))}
      </nav>

      {/* Ask and Documents stay mounted so an answer, or an upload in
          progress, survives switching pages. */}
      <main className="main">
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
            theme={theme}
          />
        )}
        {page === 'stats' && <StatsDashboard />}
      </main>
    </div>
  )
}
