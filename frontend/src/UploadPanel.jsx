import { useState, useCallback, useRef, useEffect } from 'react'

// API base — relative so Vite proxy (dev) and FastAPI static (prod) both work
const API = ''

// ── Ingested Documents Table ──────────────────────────────────────────────────

function IngestedDocuments({ refreshTrigger, onDocumentsChange }) {
  const [docs, setDocs]           = useState([])
  const [loading, setLoading]     = useState(false)
  const [deletingId, setDeleting] = useState(null)
  const [confirmId, setConfirmId] = useState(null)
  const [deleteResult, setDeleteResult]   = useState(null)
  const [purging, setPurging]             = useState(false)
  const [purgeResult, setPurgeResult]     = useState(null)

  const fetchDocs = useCallback(async () => {
    setLoading(true)
    try {
      const res  = await fetch(`${API}/documents`)
      const data = await res.json()
      const list = data.documents || []
      setDocs(list)
      onDocumentsChange?.(list)
    } catch {
      // backend may not be running yet; fail silently
    } finally {
      setLoading(false)
    }
  }, [onDocumentsChange])

  // Re-fetch whenever parent triggers (new ingest or manual refresh)
  useEffect(() => { fetchDocs() }, [fetchDocs, refreshTrigger])

  const handleDelete = async (sha256) => {
    setDeleting(sha256)
    setConfirmId(null)
    setDeleteResult(null)
    try {
      const res  = await fetch(`${API}/documents/${sha256}`, { method: 'DELETE' })
      const data = await res.json()
      setDeleteResult({ ok: true, data })
      await fetchDocs()
    } catch {
      setDeleteResult({ ok: false })
    } finally {
      setDeleting(null)
    }
  }

  const handlePurgeLegacy = async () => {
    setPurging(true)
    setPurgeResult(null)
    try {
      const res  = await fetch(`${API}/documents/purge-legacy`, { method: 'POST' })
      const data = await res.json()
      setPurgeResult({ ok: true, purged: data.purged ?? 0 })
      await fetchDocs()
    } catch {
      setPurgeResult({ ok: false })
    } finally {
      setPurging(false)
    }
  }

  const fmt = (iso) => {
    if (!iso) return '—'
    try {
      return new Date(iso).toLocaleString('en-IN', {
        day: '2-digit', month: 'short', year: 'numeric',
        hour: '2-digit', minute: '2-digit', hour12: false,
      })
    } catch { return iso }
  }

  return (
    <div style={{ marginTop: 28 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <div className="card-title" style={{ margin: 0 }}>📚 Ingested Documents</div>
        <button
          className="btn btn-ghost-danger btn-sm"
          onClick={handlePurgeLegacy}
          disabled={purging || loading}
          title="Remove stale filename-only nodes left by older versions"
          style={{ marginLeft: 'auto' }}
        >
          {purging
            ? <><span className="spinner" style={{ width: 11, height: 11 }} /> Purging…</>
            : '🧹 Purge Legacy'}
        </button>
        <button
          className="btn btn-secondary btn-sm"
          onClick={fetchDocs}
          disabled={loading}
        >
          {loading
            ? <><span className="spinner" style={{ width: 11, height: 11 }} /> Refreshing…</>
            : '↻ Refresh'}
        </button>
      </div>

      {purgeResult && (
        <div style={{
          marginBottom: 12, padding: '8px 12px', borderRadius: 8, fontSize: 12,
          background: purgeResult.ok
            ? 'rgba(63,185,80,0.10)' : 'rgba(248,81,73,0.10)',
          border: `1px solid ${purgeResult.ok
            ? 'rgba(63,185,80,0.25)' : 'rgba(248,81,73,0.25)'}`,
          color: purgeResult.ok ? 'var(--accent-green)' : 'var(--accent-red)',
        }}>
          {purgeResult.ok
            ? `🧹 Purged ${purgeResult.purged} legacy document node${purgeResult.purged !== 1 ? 's' : ''}`
            : '⚠️ Purge failed. Is the backend running?'}
        </div>
      )}

      {deleteResult && (
        <div style={{
          marginBottom: 12, padding: '8px 12px', borderRadius: 8, fontSize: 12,
          background: deleteResult.ok
            ? 'rgba(63,185,80,0.10)' : 'rgba(248,81,73,0.10)',
          border: `1px solid ${deleteResult.ok
            ? 'rgba(63,185,80,0.25)' : 'rgba(248,81,73,0.25)'}`,
          color: deleteResult.ok ? 'var(--accent-green)' : 'var(--accent-red)',
        }}>
          {deleteResult.ok
            ? `✅ Deleted "${deleteResult.data?.filename}" — Neo4j chunks: ${deleteResult.data?.neo4j?.chunks_deleted ?? '?'}, Qdrant points: ${deleteResult.data?.qdrant?.points_deleted ?? '?'}`
            : '⚠️ Delete failed. Is the backend running?'}
        </div>
      )}

      {docs.length === 0 && !loading ? (
        <div style={{
          padding: '24px', textAlign: 'center',
          border: '1px dashed var(--border)', borderRadius: 10,
          color: 'var(--text-dim)', fontSize: 13,
        }}>
          No documents ingested yet. Upload a PDF above to get started.
        </div>
      ) : (
        <div className="doc-table-wrapper">
          <table className="doc-table">
            <thead>
              <tr>
                <th>Filename</th>
                <th>Pages</th>
                <th>Entities</th>
                <th>Relations</th>
                <th>Ingested At</th>
                <th>Status</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((doc) => (
                <tr key={doc.sha256} className="doc-row">
                  <td>
                    <div className="doc-filename">
                      <span className="doc-pdf-icon">📄</span>
                      <span title={doc.filename}>{doc.filename}</span>
                    </div>
                    <div className="doc-sha">
                      {doc.sha256?.slice(0, 12)}…
                    </div>
                  </td>
                  <td><span className="doc-stat">{doc.pages ?? '—'}</span></td>
                  <td><span className="doc-stat blue">{doc.entities ?? '—'}</span></td>
                  <td><span className="doc-stat teal">{doc.relations ?? '—'}</span></td>
                  <td style={{ whiteSpace: 'nowrap', fontSize: 11, color: 'var(--text-muted)' }}>
                    {fmt(doc.ingested_at)}
                  </td>
                  <td>
                    <span className="status-badge ingested">✅ Ingested</span>
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    {confirmId === doc.sha256 ? (
                      <span style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                        <span style={{ fontSize: 11, color: 'var(--accent-orange)' }}>Confirm?</span>
                        <button
                          className="btn btn-danger btn-sm"
                          disabled={deletingId === doc.sha256}
                          onClick={() => handleDelete(doc.sha256)}
                        >
                          {deletingId === doc.sha256
                            ? <span className="spinner" style={{ width: 11, height: 11 }} />
                            : '🗑 Yes'}
                        </button>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => setConfirmId(null)}
                        >✕</button>
                      </span>
                    ) : (
                      <button
                        className="btn btn-ghost-danger btn-sm"
                        onClick={() => { setDeleteResult(null); setConfirmId(doc.sha256) }}
                        disabled={!!deletingId}
                      >
                        🗑 Delete
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


// ── Upload Panel ──────────────────────────────────────────────────────────────

export function UploadPanel({ onSuccess, onDocumentsChange }) {
  const [dragging, setDragging]   = useState(false)
  const [file, setFile]           = useState(null)
  const [loading, setLoading]     = useState(false)
  const [result, setResult]       = useState(null)
  const [error, setError]         = useState(null)
  const [linkIds, setLinkIds]     = useState(false)
  const [validFrom, setValidFrom] = useState('')
  const [validUntil, setValidUntil] = useState('')
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const inputRef = useRef()

  const handleDrop = useCallback(e => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f?.name.endsWith('.pdf')) setFile(f)
  }, [])

  const handleIngest = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    const form = new FormData()
    form.append('file', file)
    const params = new URLSearchParams({ provider: 'auto', link_ids: linkIds })
    if (validFrom)  params.append('valid_from',  validFrom)
    if (validUntil) params.append('valid_until', validUntil)
    try {
      const res = await fetch(`${API}/ingest?${params}`, { method: 'POST', body: form })
      const data = await res.json()
      setResult(data)
      if (data.status === 'ok' || data.status === 'already_ingested') {
        onSuccess?.(data)
        setRefreshTrigger(n => n + 1)  // trigger document list refresh
      }
    } catch {
      setError('Failed to connect to API. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      <div className="card">
        <div className="card-title">📄 Upload Medical PDF</div>

        <div
          className={`drop-zone ${dragging ? 'dragging' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current.click()}
        >
          <div className="drop-zone-icon">{file ? '📋' : '☁️'}</div>
          <div className="drop-zone-text">
            {file
              ? <><strong style={{ color: 'var(--text-primary)' }}>{file.name}</strong> — {(file.size / 1024).toFixed(0)} KB</>
              : <>Drop a PDF here or <span>click to browse</span></>
            }
          </div>
          <input ref={inputRef} type="file" accept=".pdf" style={{ display: 'none' }}
            onChange={e => setFile(e.target.files[0])} />
        </div>

        <div className="ingest-options">
          <label className="toggle-row">
            <input type="checkbox" checked={linkIds} onChange={e => setLinkIds(e.target.checked)} />
            Link UMLS/MeSH/RxNorm IDs (Phase 2)
          </label>
          <label className="toggle-row">
            Valid from:
            <input className="date-input" type="date" value={validFrom} onChange={e => setValidFrom(e.target.value)} />
          </label>
          <label className="toggle-row">
            Until:
            <input className="date-input" type="date" value={validUntil} onChange={e => setValidUntil(e.target.value)} />
          </label>
        </div>

        <div style={{ marginTop: 16 }}>
          <button className="btn btn-primary" onClick={handleIngest} disabled={!file || loading}>
            {loading ? <><span className="spinner" /> Processing…</> : '⚡ Ingest PDF'}
          </button>
        </div>

        {loading && <div className="progress-bar" style={{ marginTop: 12 }}><div className="progress-bar-fill" /></div>}

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
          <div style={{ marginTop: 16 }}>
            {result.status === 'already_ingested' ? (
              <div style={{
                padding: '10px 14px',
                background: 'rgba(210,153,34,0.1)',
                border: '1px solid rgba(210,153,34,0.3)',
                borderRadius: 8, fontSize: 13, color: 'var(--accent-orange)',
              }}>
                ⚡ Already ingested — {result.file}
              </div>
            ) : (
              <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(4,1fr)' }}>
                {[
                  ['Pages',     result.pages],
                  ['Chunks',    result.chunks],
                  ['Entities',  result.entities],
                  ['Relations', result.relations],
                ].map(([label, val]) => (
                  <div key={label} className="stat-card">
                    <div className="stat-value">{val}</div>
                    <div className="stat-label">{label}</div>
                  </div>
                ))}
              </div>
            )}
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
              {result.latency_ms}ms · {result.sha256}
            </div>
          </div>
        )}

        {/* ── Ingested Documents Registry ──────────────────────────── */}
        <IngestedDocuments
          refreshTrigger={refreshTrigger}
          onDocumentsChange={onDocumentsChange}
        />
      </div>
    </div>
  )
}
