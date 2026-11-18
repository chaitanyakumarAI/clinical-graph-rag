import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList } from 'recharts'
import { Icon } from './Icon'
import { ENTITY_TYPES, typeVar, humanType } from './entityTypes'

// API base — relative so Vite proxy (dev) and FastAPI static (prod) both work
const API = ''

const relName = r => r.toLowerCase().replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase())

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="t-name">
        <span style={{ width: 8, height: 8, borderRadius: 2, background: d.color }} />{d.label}
      </div>
      <div className="t-value num">{d.count.toLocaleString()}</div>
    </div>
  )
}

// Horizontal bars, largest first, value at the tip. A Table view carries the
// same numbers for anyone who cannot rely on the bars.
function BarCard({ title, sub, rows }) {
  const [view, setView] = useState('chart')
  const height = rows.length * 30 + 8
  return (
    <div className="card chart-card">
      <div className="card-head">
        <div>
          <div className="card-title">{title}</div>
          <div className="card-sub">{sub}</div>
        </div>
        <div className="view-switch" role="group" aria-label={`${title} view`}>
          <button type="button" aria-pressed={view === 'chart'} onClick={() => setView('chart')}>Chart</button>
          <button type="button" aria-pressed={view === 'table'} onClick={() => setView('table')}>Table</button>
        </div>
      </div>
      <div className="card-pad">
        {view === 'chart' ? (
          <ResponsiveContainer width="100%" height={height}>
            <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 44, left: 0, bottom: 0 }} barCategoryGap={8}>
              <XAxis type="number" hide domain={[0, 'dataMax']} />
              <YAxis type="category" dataKey="label" width={128} axisLine={false} tickLine={false}
                tick={{ fontSize: 12.5, fill: 'var(--text-2)' }} interval={0} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: 'var(--surface-2)' }} />
              <Bar dataKey="count" barSize={16} radius={[0, 4, 4, 0]} isAnimationActive={false}>
                {rows.map(d => <Cell key={d.label} fill={d.color} />)}
                <LabelList dataKey="count" position="right" offset={8}
                  style={{ fontSize: 12, fill: 'var(--text-2)', fontVariantNumeric: 'tabular-nums' }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <table className="mini-table">
            <tbody>
              {rows.map(d => <tr key={d.label}><td>{d.label}</td><td>{d.count.toLocaleString()}</td></tr>)}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

export function StatsDashboard() {
  const [stats, setStats]       = useState(null)
  const [docs, setDocs]         = useState([])
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [reload, setReload]     = useState(0)

  useEffect(() => {
    const getJson = url => fetch(`${API}${url}`).then(r => {
      if (!r.ok) throw new Error(`${url} returned HTTP ${r.status}`)
      return r.json()
    })
    setLoading(true)
    setError(null)
    // allSettled: one failing endpoint should not blank the whole page
    Promise.allSettled([getJson('/graph/stats'), getJson('/documents')])
      .then(([s, d]) => {
        setStats(s.status === 'fulfilled' ? s.value : null)
        setDocs(d.status === 'fulfilled' ? d.value.documents || [] : [])
        const failed = [s, d].filter(r => r.status === 'rejected')
        if (failed.length === 2) setError('Cannot reach the API. Is the backend running on port 8000?')
        else if (failed.length) setError(`Some data could not be loaded: ${failed.map(f => f.reason.message).join('; ')}`)
      })
      .finally(() => setLoading(false))
  }, [reload])

  const nodeRows = stats ? Object.entries(stats.nodes).filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => ({ label: humanType(k), count: v, color: ENTITY_TYPES[k] ? typeVar(k) : 'var(--series-other)' })) : []
  const edgeRows = stats ? Object.entries(stats.edges).filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => ({ label: relName(k), count: v, color: 'var(--accent)' })) : []
  const totalNodes = nodeRows.reduce((s, d) => s + d.count, 0)
  const totalEdges = edgeRows.reduce((s, d) => s + d.count, 0)
  const totalPages = docs.reduce((s, d) => s + (d.pages || 0), 0)

  const tiles = [
    ['Entities',  totalNodes, 'graph',    `${nodeRows.length} types`],
    ['Relations', totalEdges, 'link',     `${edgeRows.length} types`],
    ['Documents', docs.length, 'file',    'in the library'],
    ['Pages',     totalPages, 'layers',   'read and indexed'],
  ]

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Overview</h1>
          <p className="page-desc">What the knowledge graph holds right now.</p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn btn-secondary" onClick={() => setReload(n => n + 1)} disabled={loading}>
            {loading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Icon name="refresh" size={15} />}Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="alert error" role="alert" style={{ marginBottom: 20 }}>
          <Icon name="alert" size={16} /><div className="alert-body">{error}</div>
        </div>
      )}

      <div className="tiles">
        {tiles.map(([label, value, icon, note]) => (
          <div key={label} className="tile">
            <div className="tile-label"><Icon name={icon} size={15} />{label}</div>
            <div className="tile-value">{loading && !stats ? <span className="skeleton" style={{ display: 'block', width: 60, height: 26 }} /> : value.toLocaleString()}</div>
            <div className="tile-note">{note}</div>
          </div>
        ))}
      </div>

      {(nodeRows.length > 0 || edgeRows.length > 0) && (
        <div className="overview-grid">
          {nodeRows.length > 0 && <BarCard title="Entities by type" sub="Colours match the graph" rows={nodeRows} />}
          {edgeRows.length > 0 && <BarCard title="Relations by type" sub="Directed edges between entities" rows={edgeRows} />}
        </div>
      )}

      {docs.length > 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-head">
            <span className="card-title">By document</span>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th>Document</th><th className="num">Pages</th><th className="num">Entities</th><th className="num">Relations</th></tr>
              </thead>
              <tbody>
                {docs.map(doc => (
                  <tr key={doc.sha256}>
                    <td>
                      <div className="doc-name">
                        <span className="doc-icon"><Icon name="file" size={16} /></span>
                        <span className="doc-title">{doc.filename}</span>
                      </div>
                    </td>
                    <td className="num">{doc.pages}</td>
                    <td className="num">{doc.entities}</td>
                    <td className="num">{doc.relations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!loading && !error && docs.length === 0 && totalNodes === 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="empty">
            <div className="empty-icon"><Icon name="file" size={22} /></div>
            <div className="empty-title">The graph is empty</div>
            <div>Upload a PDF on the Documents page to get started.</div>
          </div>
        </div>
      )}
    </div>
  )
}
