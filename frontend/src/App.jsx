import { useState, useCallback } from 'react'
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

export default function App() {
  const [page, setPage]                   = useState('query')
  const [highlightEntities, setHighlight] = useState([])
  // Shared document list — fetched in UploadPanel, used by GraphExplorer filter
  const [documents, setDocuments]         = useState([])

  function handleQueryResult(result) {
    setHighlight(result.linked_entities || [])
  }

  const handleDocumentsChange = useCallback((docs) => {
    setDocuments(docs)
  }, [])

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="header">
        <div className="header-logo">🧬</div>
        <span className="header-title">Medical Graph RAG</span>
        <span className="header-subtitle">v2.0</span>
        <div className="header-badge">
          <span className="badge">Neo4j</span>
          <span className="badge">Qdrant</span>
          <span className="badge green">Phase 1–6</span>
        </div>
      </header>

      {/* Sidebar */}
      <nav className="sidebar">
        {PAGES.map(p => (
          <div
            key={p.id}
            className={`nav-item ${page === p.id ? 'active' : ''}`}
            onClick={() => setPage(p.id)}
          >
            <span className="nav-icon">{p.icon}</span>
            {p.label}
          </div>
        ))}

        {/* Document count pill in sidebar */}
        {documents.length > 0 && (
          <div
            className="nav-item"
            onClick={() => setPage('upload')}
            style={{ cursor: 'pointer', marginTop: 4 }}
          >
            <span className="nav-icon">📚</span>
            <span style={{ fontSize: 12 }}>{documents.length} doc{documents.length !== 1 ? 's' : ''} ingested</span>
          </div>
        )}

        <div style={{ marginTop: 'auto', padding: '16px 20px',
          borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', lineHeight: 1.6 }}>
            <div>🔐 Qdrant-first pipeline</div>
            <div>🧮 Dynamic Cypher (Ph.3)</div>
            <div>📊 GDS Algorithms (Ph.4)</div>
            <div>🔗 UMLS/MeSH IDs (Ph.2)</div>
          </div>
        </div>
      </nav>

      {/* Main */}
      <main className="main-content">
        {page === 'query' && (
          <QueryPanel onQueryResult={handleQueryResult} />
        )}
        {page === 'upload' && (
          <UploadPanel
            onSuccess={() => {}}
            onDocumentsChange={handleDocumentsChange}
          />
        )}
        {page === 'graph' && (
          <GraphExplorer
            highlightEntities={highlightEntities}
            documents={documents}
          />
        )}
        {page === 'stats' && <StatsDashboard />}
      </main>
    </div>
  )
}
