import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

// API base — relative so Vite proxy (dev) and FastAPI static (prod) both work
const API = ''

const NODE_COLORS = {
  Disease:'#f85149', Drug:'#388bfd', Gene:'#3fb950', Protein:'#39d0d8',
  Pathway:'#d29922', Symptom:'#bc8cff', Treatment:'#58a6ff', SideEffect:'#ff7b72',
  Organ:'#e3b341', Compound:'#56d364', PatientGroup:'#ffa657',
}
const EDGE_COLORS = {
  TREATS:'#388bfd', CAUSES:'#f85149', TARGETS:'#3fb950', REGULATES:'#39d0d8',
  INHIBITS:'#ff7b72', ACTIVATES:'#56d364', HAS_SYMPTOM:'#bc8cff',
  INTERACTS_WITH:'#d29922', BIOMARKER_FOR:'#ffa657', RELATED_TO:'#484f58',
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{background:'var(--bg-card)', border:'1px solid var(--border)',
      borderRadius:8, padding:'8px 12px', fontSize:12}}>
      <div style={{fontWeight:600, marginBottom:4}}>{label}</div>
      <div style={{color:'var(--accent-blue)'}}>{payload[0].value.toLocaleString()}</div>
    </div>
  )
}

export function StatsDashboard() {
  const [stats, setStats]       = useState(null)
  const [docs, setDocs]         = useState([])
  const [health, setHealth]     = useState(null)
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
    Promise.allSettled([getJson('/graph/stats'), getJson('/documents'), getJson('/health')])
      .then(([s, d, h]) => {
        setStats(s.status === 'fulfilled' ? s.value : null)
        setDocs(d.status === 'fulfilled' ? d.value.documents || [] : [])
        setHealth(h.status === 'fulfilled' ? h.value : null)
        const failed = [s, d, h].filter(r => r.status === 'rejected')
        if (failed.length === 3) setError('Cannot reach the API. Is the backend running on port 8000?')
        else if (failed.length) setError(`Some data could not be loaded: ${failed.map(f => f.reason.message).join('; ')}`)
      })
      .finally(() => setLoading(false))
  }, [reload])

  if (loading) return (
    <div className="card" style={{textAlign:'center', padding:60}}>
      <span className="spinner" style={{width:24,height:24}} />
      <div style={{marginTop:12, color:'var(--text-muted)'}}>Loading stats…</div>
    </div>
  )

  const nodeData  = stats ? Object.entries(stats.nodes).filter(([,v]) => v > 0)
    .map(([k,v]) => ({name:k, count:v})) : []
  const edgeData  = stats ? Object.entries(stats.edges).filter(([,v]) => v > 0)
    .map(([k,v]) => ({name:k, count:v})) : []
  const totalNodes = nodeData.reduce((s,d) => s + d.count, 0)
  const totalEdges = edgeData.reduce((s,d) => s + d.count, 0)

  const apiStatus = !health ? '❌ Offline'
    : health.status === 'ok' ? '✅ Online' : '⚠️ Degraded'

  return (
    <div style={{display:'flex', flexDirection:'column', gap:20}}>

      {error && (
        <div className="alert error" role="alert" style={{marginTop:0, display:'flex', alignItems:'center', gap:12}}>
          <span>⚠️ {error}</span>
          <button className="btn btn-secondary btn-sm" style={{marginLeft:'auto'}}
            onClick={() => setReload(n => n + 1)}>↻ Retry</button>
        </div>
      )}

      {/* Top stats */}
      <div className="stats-grid" style={{gridTemplateColumns:'repeat(4,1fr)'}}>
        {[
          ['Total Nodes', totalNodes.toLocaleString()],
          ['Total Edges', totalEdges.toLocaleString()],
          ['Documents',   docs.length],
          ['API Status',  apiStatus],
        ].map(([label, val]) => (
          <div key={label} className="stat-card">
            <div className="stat-value" style={{fontSize:20}}>{val}</div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>

      {/* Node distribution chart */}
      {nodeData.length > 0 && (
        <div className="card">
          <div className="card-title">Entity Distribution</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={nodeData} margin={{top:0, right:0, left:-10, bottom:0}}>
              <XAxis dataKey="name" tick={{fontSize:10, fill:'#7d8590'}} />
              <YAxis tick={{fontSize:10, fill:'#7d8590'}} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[4,4,0,0]}>
                {nodeData.map(d => (
                  <Cell key={d.name} fill={NODE_COLORS[d.name] || '#388bfd'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Edge distribution chart */}
      {edgeData.length > 0 && (
        <div className="card">
          <div className="card-title">Relationship Distribution</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={edgeData} margin={{top:0, right:0, left:-10, bottom:0}}>
              <XAxis dataKey="name" tick={{fontSize:9, fill:'#7d8590'}} />
              <YAxis tick={{fontSize:10, fill:'#7d8590'}} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[4,4,0,0]}>
                {edgeData.map(d => (
                  <Cell key={d.name} fill={EDGE_COLORS[d.name] || '#388bfd'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Document registry */}
      {docs.length > 0 && (
        <div className="card">
          <div className="card-title">📚 Ingested Documents</div>
          <div style={{display:'flex', flexDirection:'column', gap:8}}>
            {docs.map(doc => (
              <div key={doc.sha256}
                style={{display:'flex', alignItems:'center', gap:12,
                  padding:'10px 14px', background:'var(--bg-input)',
                  borderRadius:8, border:'1px solid var(--border)'}}>
                <div style={{fontSize:20}}>📄</div>
                <div style={{flex:1}}>
                  <div style={{fontSize:13, fontWeight:600}}>{doc.filename}</div>
                  <div style={{fontSize:11, color:'var(--text-muted)', marginTop:2}}>
                    {doc.pages} pages · {doc.entities} entities · {doc.relations} relations
                    · {new Date(doc.ingested_at).toLocaleDateString()}
                  </div>
                </div>
                <div style={{display:'flex', gap:6}}>
                  <span className="badge">{doc.entities} ent.</span>
                  <span className="badge green">{doc.relations} rel.</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {docs.length === 0 && !loading && !error && (
        <div className="card" style={{textAlign:'center', padding:40, color:'var(--text-muted)'}}>
          No documents ingested yet. Upload a PDF to get started.
        </div>
      )}
    </div>
  )
}
