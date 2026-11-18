import { useState, useCallback, useRef, useEffect } from 'react'
import { Icon } from './Icon'

// API base — relative so Vite proxy (dev) and FastAPI static (prod) both work
const API = ''

const fmtDate = iso => {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString(undefined, {
      day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch { return iso }
}

// ── Ingested documents ────────────────────────────────────────────────────────

function IngestedDocuments({ refreshTrigger, onDocumentsChange }) {
  const [docs, setDocs]           = useState([])
  const [loading, setLoading]     = useState(false)
  const [loadError, setLoadError] = useState(false)
  const [deletingId, setDeleting] = useState(null)
  const [confirmId, setConfirmId] = useState(null)
  const [notice, setNotice]       = useState(null)   // { kind, text }
  const [purging, setPurging]     = useState(false)

  const fetchDocs = useCallback(async () => {
    setLoading(true)
    try {
      const res  = await fetch(`${API}/documents`)
      const data = await res.json()
      const list = data.documents || []
      setDocs(list)
      setLoadError(false)
      onDocumentsChange?.(list)
    } catch {
      setLoadError(true)
    } finally {
      setLoading(false)
    }
  }, [onDocumentsChange])

  // Re-fetch whenever parent triggers (new ingest or manual refresh)
  useEffect(() => { fetchDocs() }, [fetchDocs, refreshTrigger])

  const handleDelete = async (sha256) => {
    setDeleting(sha256)
    setConfirmId(null)
    setNotice(null)
    try {
      const res  = await fetch(`${API}/documents/${sha256}`, { method: 'DELETE' })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail)
      setNotice({ kind: 'good', text: <>Deleted <strong>{data.filename}</strong>: {data.neo4j?.chunks_deleted ?? '?'} graph chunks and {data.qdrant?.points_deleted ?? '?'} vectors removed.</> })
      await fetchDocs()
    } catch {
      setNotice({ kind: 'error', text: 'Delete failed. Is the backend running?' })
    } finally {
      setDeleting(null)
    }
  }

  const handlePurgeLegacy = async () => {
    setPurging(true)
    setNotice(null)
    try {
      const res  = await fetch(`${API}/documents/purge-legacy`, { method: 'POST' })
      const data = await res.json()
      const n = data.purged ?? 0
      setNotice({ kind: 'good', text: `Removed ${n} legacy document node${n !== 1 ? 's' : ''}.` })
      await fetchDocs()
    } catch {
      setNotice({ kind: 'error', text: 'Clean-up failed. Is the backend running?' })
    } finally {
      setPurging(false)
    }
  }

  return (
    <div className="card" style={{ marginTop: 20 }}>
      <div className="card-head">
        <span className="card-title">Library</span>
        <span className="card-sub num">{docs.length} document{docs.length !== 1 ? 's' : ''}</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <button type="button" className="btn btn-ghost btn-sm" onClick={handlePurgeLegacy}
            disabled={purging || loading}
            title="Remove stale filename-only nodes left by older versions">
            {purging ? <span className="spinner" style={{ width: 13, height: 13 }} /> : <Icon name="sparkle" size={14} />}
            Clean up legacy nodes
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={fetchDocs} disabled={loading}>
            {loading ? <span className="spinner" style={{ width: 13, height: 13 }} /> : <Icon name="refresh" size={14} />}
            Refresh
          </button>
        </div>
      </div>

      {notice && (
        <div style={{ padding: '14px 22px 0' }}>
          <div className={`alert ${notice.kind}`} role="status">
            <Icon name={notice.kind === 'good' ? 'check' : 'alert'} size={16} />
            <div className="alert-body">{notice.text}</div>
          </div>
        </div>
      )}

      {docs.length === 0 && !loading ? (
        <div className="empty">
          <div className="empty-icon"><Icon name={loadError ? 'alert' : 'file'} size={22} /></div>
          <div className="empty-title">{loadError ? 'Could not load documents' : 'No documents yet'}</div>
          <div>{loadError ? 'Is the backend running on port 8000?' : 'Upload a PDF above to build the graph.'}</div>
        </div>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Document</th>
                <th className="num">Pages</th>
                <th className="num">Entities</th>
                <th className="num">Relations</th>
                <th>Added</th>
                <th><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {docs.map(doc => (
                <tr key={doc.sha256}>
                  <td>
                    <div className="doc-name">
                      <span className="doc-icon"><Icon name="file" size={17} /></span>
                      <div style={{ minWidth: 0 }}>
                        <div className="doc-title">{doc.filename}</div>
                        <div className="doc-hash" title={doc.sha256}>{doc.sha256?.slice(0, 12)}</div>
                      </div>
                    </div>
                  </td>
                  <td className="num">{doc.pages ?? '—'}</td>
                  <td className="num">{doc.entities ?? '—'}</td>
                  <td className="num">{doc.relations ?? '—'}</td>
                  <td style={{ whiteSpace: 'nowrap', color: 'var(--text-2)' }}>{fmtDate(doc.ingested_at)}</td>
                  <td>
                    {confirmId === doc.sha256 ? (
                      <span className="confirm">
                        <span className="confirm-text">Delete from graph and index?</span>
                        <button type="button" className="btn btn-danger btn-sm"
                          disabled={deletingId === doc.sha256}
                          onClick={() => handleDelete(doc.sha256)}>
                          {deletingId === doc.sha256 ? <span className="spinner" style={{ width: 13, height: 13 }} /> : 'Delete'}
                        </button>
                        <button type="button" className="btn btn-ghost btn-sm" onClick={() => setConfirmId(null)}>Cancel</button>
                      </span>
                    ) : (
                      <button type="button" className="btn btn-ghost-danger btn-sm btn-icon"
                        onClick={() => { setNotice(null); setConfirmId(doc.sha256) }}
                        disabled={!!deletingId}
                        aria-label={`Delete ${doc.filename}`} title="Delete">
                        <Icon name="trash" size={16} />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ── Upload ────────────────────────────────────────────────────────────────────

export function UploadPanel({ onSuccess, onDocumentsChange }) {
  const [dragging, setDragging]   = useState(false)
  const [file, setFile]           = useState(null)
  const [loading, setLoading]     = useState(false)
  const [result, setResult]       = useState(null)
  const [error, setError]         = useState(null)
  const [linkIds, setLinkIds]     = useState(false)
  const [largePdf, setLargePdf]   = useState(false)
  const [validFrom, setValidFrom] = useState('')
  const [validUntil, setValidUntil] = useState('')
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const inputRef = useRef()

  const pick = f => { setFile(f); setError(null); setResult(null) }

  const handleDrop = useCallback(e => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (!f) return
    if (f.name.toLowerCase().endsWith('.pdf')) pick(f)
    else setError(`"${f.name}" is not a PDF.`)
  }, [])

  const handleIngest = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    const form = new FormData()
    form.append('file', file)
    const params = new URLSearchParams({
      provider: 'auto',
      link_ids: linkIds,
      // Bigger chunks = half as many LLM calls, which matters on free-tier rate limits
      chunk_tokens: largePdf ? 1024 : 512,
    })
    if (validFrom)  params.append('valid_from',  validFrom)
    if (validUntil) params.append('valid_until', validUntil)
    try {
      const res = await fetch(`${API}/ingest?${params}`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) {
        // e.g. 429 = LLM rate limit used up (progress is cached — retry later)
        setError(data.detail || `Ingestion failed (HTTP ${res.status})`)
        return
      }
      setResult(data)
      if (data.status === 'ok' || data.status === 'already_ingested') {
        onSuccess?.(data)
        setRefreshTrigger(n => n + 1)  // trigger document list refresh
      }
    } catch {
      setError('Could not reach the API. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Documents</h1>
          <p className="page-desc">
            Add a medical PDF. Its text is split into chunks, embedded into Qdrant, and the
            entities and relations an LLM finds are written to the Neo4j graph.
          </p>
        </div>
      </div>

      <div className="card upload-grid">
        <div>
          <div
            className={`drop-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current.click()}
            onKeyDown={e => {
              if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inputRef.current.click() }
            }}
            role="button"
            tabIndex={0}
            aria-label={file ? `Selected ${file.name}. Choose a different PDF` : 'Choose a PDF to upload'}
          >
            <span className="drop-icon"><Icon name={file ? 'file' : 'upload'} size={22} /></span>
            {file ? (
              <>
                <div className="drop-title" style={{ overflowWrap: 'anywhere' }}>{file.name}</div>
                <div className="drop-sub">{(file.size / 1024).toFixed(0)} KB · <span className="link">choose another</span></div>
              </>
            ) : (
              <>
                <div className="drop-title">Drop a PDF here</div>
                <div className="drop-sub">or <span className="link">browse your files</span></div>
              </>
            )}
            <input ref={inputRef} type="file" accept=".pdf" style={{ display: 'none' }}
              onChange={e => { if (e.target.files[0]) pick(e.target.files[0]) }} />
          </div>
        </div>

        <div>
          <div className="section-label" style={{ marginBottom: 0 }}>Options</div>
          <label className="switch">
            <input type="checkbox" role="switch" checked={largePdf} onChange={e => setLargePdf(e.target.checked)} />
            <span className="switch-text">Large PDF mode
              <span className="switch-hint">1024-token chunks: half the LLM calls, fewer rate-limit pauses</span></span>
          </label>
          <label className="switch">
            <input type="checkbox" role="switch" checked={linkIds} onChange={e => setLinkIds(e.target.checked)} />
            <span className="switch-text">Link UMLS, MeSH and RxNorm IDs
              <span className="switch-hint">Looks each entity up in public vocabularies (slower)</span></span>
          </label>
          <div>
            <div className="field-label" style={{ marginBottom: 6 }}>Valid between <span className="muted">(optional)</span></div>
            <div className="date-row">
              <input className="input" type="date" aria-label="Valid from" value={validFrom} onChange={e => setValidFrom(e.target.value)} />
              <input className="input" type="date" aria-label="Valid until" value={validUntil} onChange={e => setValidUntil(e.target.value)} />
            </div>
          </div>

          <button type="button" className="btn btn-primary" style={{ height: 40, marginTop: 'auto' }}
            onClick={handleIngest} disabled={!file || loading}>
            {loading ? <><span className="spinner" style={{ width: 14, height: 14 }} />Ingesting… this can take a few minutes</> : <><Icon name="upload" size={16} />Ingest PDF</>}
          </button>
          {loading && <div className="progress" aria-hidden="true"><span /></div>}

          {error && (
            <div className="alert error" role="alert">
              <Icon name="alert" size={16} /><div className="alert-body">{error}</div>
            </div>
          )}

          {result?.status === 'already_ingested' && (
            <div className="alert warn" role="status">
              <Icon name="alert" size={16} /><div className="alert-body"><strong>{result.file}</strong> is already in the library.</div>
            </div>
          )}
          {result && result.status !== 'already_ingested' && (
            <div>
              <div className="alert good" role="status" style={{ marginBottom: 10 }}>
                <Icon name="check" size={16} />
                <div className="alert-body">Ingested in {(result.latency_ms / 1000).toFixed(1)}s</div>
              </div>
              <div className="tiles compact">
                {[['Pages', result.pages], ['Chunks', result.chunks], ['Entities', result.entities], ['Relations', result.relations]]
                  .map(([label, val]) => (
                    <div key={label} className="tile">
                      <div className="tile-label">{label}</div>
                      <div className="tile-value num">{val}</div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <IngestedDocuments refreshTrigger={refreshTrigger} onDocumentsChange={onDocumentsChange} />
    </div>
  )
}
