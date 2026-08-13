import { useEffect, useRef, useState, useCallback } from 'react'
import cytoscape from 'cytoscape'

// API base — relative so Vite proxy (dev) and FastAPI static (prod) both work
const API = ''

// Entity type → color mapping
const LABEL_COLORS = {
  Disease:           '#f85149',
  Drug:              '#388bfd',
  Gene:              '#3fb950',
  Protein:           '#39d0d8',
  Pathway:           '#d29922',
  Symptom:           '#bc8cff',
  Treatment:         '#58a6ff',
  SideEffect:        '#ff7b72',
  PatientGroup:      '#ffa657',
  Organ:             '#e3b341',
  MedicalGuideline:  '#a5d6ff',
  Compound:          '#56d364',
  CellType:          '#f0883e',
  MolecularFunction: '#79c0ff',
  BiologicalProcess: '#d2a8ff',
  Document:          '#484f58',
}

function buildCyElements(nodes, edges) {
  const els = []
  nodes.forEach(n => {
    if (n.name) els.push({
      data: {
        id: n.name, label: n.name, nodeType: n.label || 'Unknown',
        confidence: n.confidence, stage: n.stage,
        sources: n.sources, pages: n.pages,
        umls: n.umls_cui, mesh: n.mesh_id, rxnorm: n.rxnorm_id,
      }
    })
  })
  edges.forEach((e, i) => {
    if (e.source && e.target) els.push({
      data: {
        id: `e${i}`, source: e.source, target: e.target,
        label: e.relation, confidence: e.confidence,
        pages: e.pages, sources: e.sources,
      }
    })
  })
  return els
}

function buildCyStyle() {
  const nodeStyles = Object.entries(LABEL_COLORS).map(([label, color]) => ({
    selector: `node[nodeType="${label}"]`,
    style: { 'background-color': color }
  }))
  return [
    {
      selector: 'node',
      style: {
        'label': 'data(label)',
        'font-size': '10px',
        'color': '#e6edf3',
        'text-valign': 'bottom',
        'text-margin-y': '4px',
        'text-wrap': 'truncate',
        'text-max-width': '90px',
        'width': 28, 'height': 28,
        'background-color': '#388bfd',
        'border-width': 2,
        'border-color': '#30363d',
        'transition-property': 'background-color, border-color, width, height, opacity',
        'transition-duration': '0.15s',
      }
    },
    {
      selector: 'node:selected',
      style: {
        'border-color': '#ffffff',
        'border-width': 3,
        'width': 36, 'height': 36,
      }
    },
    {
      selector: 'edge',
      style: {
        'label': 'data(label)',
        'font-size': '8px',
        'color': '#7d8590',
        'text-rotation': 'autorotate',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#30363d',
        'line-color': '#30363d',
        'width': 1.5,
        'arrow-scale': 0.8,
      }
    },
    {
      selector: 'edge:selected',
      style: { 'line-color': '#388bfd', 'target-arrow-color': '#388bfd', 'width': 2.5 }
    },
    ...nodeStyles,
  ]
}

const LABELS = [
  'Disease', 'Drug', 'Gene', 'Protein', 'Pathway', 'Symptom', 'Treatment',
  'SideEffect', 'Organ', 'Compound', 'CellType', 'MolecularFunction',
  'BiologicalProcess', 'Document',
]

// ── Main component ────────────────────────────────────────────────────────────

export function GraphExplorer({ highlightEntities = [], documents = [] }) {
  const cyRef       = useRef(null)
  const cyInstance  = useRef(null)
  const [loading, setLoading]         = useState(false)
  const [selectedNode, setSelected]   = useState(null)
  const [labelFilter, setLabelFilter] = useState('')
  const [minConf, setMinConf]         = useState(0)
  const [nodeDetail, setNodeDetail]   = useState(null)
  // Document source filter
  const [sourceDoc, setSourceDoc]     = useState('')   // filename to highlight
  const [allNodes, setAllNodes]       = useState([])   // cached full node list
  const [allEdges, setAllEdges]       = useState([])

  const fetchAndRender = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({ limit: 80, min_conf: minConf })
      if (labelFilter) params.append('label', labelFilter)
      const res  = await fetch(`${API}/graph/explore?${params}`)
      const data = await res.json()
      const nodes = data.nodes || []
      const edges = data.edges || []
      setAllNodes(nodes)
      setAllEdges(edges)
      renderGraph(nodes, edges, sourceDoc)
    } catch (e) {
      console.error('[graph] fetch failed:', e)
    } finally {
      setLoading(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [labelFilter, minConf])

  useEffect(() => { fetchAndRender() }, [fetchAndRender])

  // Re-apply source filter when it changes without a full refetch
  useEffect(() => {
    if (allNodes.length) renderGraph(allNodes, allEdges, sourceDoc)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceDoc])

  // Highlight subgraph after a query result
  useEffect(() => {
    if (!highlightEntities.length || !cyInstance.current) return
    const cy = cyInstance.current
    cy.nodes().style({ opacity: 0.2 })
    cy.edges().style({ opacity: 0.1 })
    highlightEntities.forEach(name => {
      const node = cy.$(`#${CSS.escape(name)}`)
      if (node.length) {
        node.style({ opacity: 1 })
        node.neighborhood().style({ opacity: 0.7 })
      }
    })
  }, [highlightEntities])

  function renderGraph(nodes, edges, docFilter = '') {
    if (!cyRef.current) return
    if (cyInstance.current) cyInstance.current.destroy()

    const cy = cytoscape({
      container: cyRef.current,
      elements:  buildCyElements(nodes, edges),
      style:     buildCyStyle(),
      layout:    { name: 'cose', animate: true, padding: 30,
                   nodeRepulsion: 4500, gravity: 0.15 },
      zoom: 1, pan: { x: 0, y: 0 },
      minZoom: 0.3, maxZoom: 4,
    })

    // Apply document source dimming
    if (docFilter) {
      cy.nodes().forEach(node => {
        const sources = node.data('sources') || []
        const inDoc = Array.isArray(sources)
          ? sources.some(s => s === docFilter || s?.includes?.(docFilter))
          : false
        node.style({ opacity: inDoc ? 1 : 0.12 })
      })
      cy.edges().forEach(edge => {
        const srcOpacity = edge.source().style('opacity')
        const tgtOpacity = edge.target().style('opacity')
        edge.style({ opacity: srcOpacity > 0.5 && tgtOpacity > 0.5 ? 0.9 : 0.06 })
      })
    }

    cy.on('tap', 'node', async evt => {
      const node = evt.target
      setSelected({ name: node.data('label'), ...node.data() })
      // Fetch full provenance from backend
      try {
        const res  = await fetch(`${API}/graph/node/${encodeURIComponent(node.data('label'))}`)
        const data = await res.json()
        setNodeDetail(data)
      } catch { setNodeDetail(null) }
    })

    cy.on('tap', function(evt) {
      if (evt.target === cy) { setSelected(null); setNodeDetail(null) }
    })

    cyInstance.current = cy
  }

  // Count how many nodes are visible for the selected source doc
  const visibleCount = sourceDoc
    ? allNodes.filter(n => {
        const s = n.sources || []
        return Array.isArray(s) ? s.some(x => x === sourceDoc || x?.includes?.(sourceDoc)) : false
      }).length
    : allNodes.length

  return (
    <div className="card">
      <div className="card-title">🕸️ Knowledge Graph Explorer</div>

      {/* ── Controls row ─────────────────────────────────────────────── */}
      <div className="graph-controls">
        {/* Entity type filter */}
        <select className="select-input" value={labelFilter} onChange={e => setLabelFilter(e.target.value)}>
          <option value="">All Entity Types</option>
          {LABELS.map(l => <option key={l} value={l}>{l}</option>)}
        </select>

        {/* Min confidence slider */}
        <label className="toggle-row">
          Min confidence:
          <input type="range" min="0" max="1" step="0.05" value={minConf}
            onChange={e => setMinConf(parseFloat(e.target.value))}
            style={{ width: 80, accentColor: 'var(--accent-blue)' }} />
          <span style={{ fontSize: 12 }}>{minConf.toFixed(2)}</span>
        </label>

        <button className="btn btn-secondary btn-sm" onClick={fetchAndRender} disabled={loading}>
          {loading ? <span className="spinner" style={{ width: 12, height: 12 }} /> : '↻'} Refresh
        </button>

        {/* Legend */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginLeft: 'auto' }}>
          {['Disease', 'Drug', 'Gene', 'Protein', 'Pathway'].map(l => (
            <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--text-muted)' }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%',
                background: LABEL_COLORS[l], display: 'inline-block' }} />
              {l}
            </span>
          ))}
        </div>
      </div>

      {/* ── Document Source Filter ────────────────────────────────────── */}
      <div className="graph-doc-filter">
        <span className="graph-doc-filter-label">📄 Filter by document:</span>
        <select
          className="select-input"
          value={sourceDoc}
          onChange={e => setSourceDoc(e.target.value)}
          style={{ minWidth: 200, maxWidth: 320 }}
        >
          <option value="">Show All Documents</option>
          {documents.map(d => (
            <option key={d.sha256} value={d.filename}>{d.filename}</option>
          ))}
        </select>
        {sourceDoc && (
          <>
            <span className="graph-doc-filter-count">
              {visibleCount} node{visibleCount !== 1 ? 's' : ''} from this PDF
            </span>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => setSourceDoc('')}
            >✕ Clear</button>
          </>
        )}
      </div>

      {/* ── Graph canvas + node detail ────────────────────────────────── */}
      <div style={{ position: 'relative' }}>
        <div id="cy-container" ref={cyRef} />

        {selectedNode && (
          <div className="node-detail-panel">
            <div style={{
              fontWeight: 700, fontSize: 13, marginBottom: 12,
              color: LABEL_COLORS[selectedNode.nodeType] || 'var(--text-primary)',
            }}>
              {selectedNode.name}
              <span style={{ marginLeft: 8, fontSize: 11, fontWeight: 400,
                color: 'var(--text-muted)' }}>{selectedNode.nodeType}</span>
            </div>

            {[
              ['Confidence', selectedNode.confidence?.toFixed?.(3)],
              ['Stage',      selectedNode.stage],
              ['UMLS CUI',   selectedNode.umls],
              ['MeSH ID',    selectedNode.mesh],
              ['RxNorm ID',  selectedNode.rxnorm],
            ].filter(([, v]) => v).map(([k, v]) => (
              <div key={k} className="detail-row">
                <div className="detail-label">{k}</div>
                <div className="detail-value">{v}</div>
              </div>
            ))}

            {/* Pages */}
            {selectedNode.pages?.length > 0 && (
              <div className="detail-row">
                <div className="detail-label">Pages</div>
                <div className="detail-value">
                  {selectedNode.pages.map(p => (
                    <span key={p} className="page-badge">p.{p}</span>
                  ))}
                </div>
              </div>
            )}

            {/* Source documents */}
            {selectedNode.sources?.length > 0 && (
              <div style={{ marginTop: 10 }}>
                <div style={{
                  fontSize: 10, fontWeight: 600, color: 'var(--text-dim)',
                  textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 6,
                }}>Source Documents</div>
                {selectedNode.sources.map((s, i) => (
                  <div key={i} style={{
                    fontSize: 11, padding: '3px 7px', marginBottom: 4,
                    background: 'var(--bg-input)', borderRadius: 5,
                    color: 'var(--accent-green)',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }} title={s}>📄 {s}</div>
                ))}
              </div>
            )}

            {/* Relationships */}
            {nodeDetail?.relationships?.length > 0 && (
              <>
                <div style={{
                  borderTop: '1px solid var(--border)', marginTop: 12, paddingTop: 12,
                  fontSize: 11, fontWeight: 600, color: 'var(--text-muted)',
                  textTransform: 'uppercase', letterSpacing: '0.6px', marginBottom: 8,
                }}>
                  Relationships ({nodeDetail.relationships.length})
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {nodeDetail.relationships.slice(0, 8).map((r, i) => (
                    <div key={i} style={{
                      fontSize: 11, padding: '5px 8px',
                      background: 'var(--bg-input)', borderRadius: 5,
                    }}>
                      <span style={{ color: 'var(--accent-teal)' }}>{r.relation}</span>
                      {' → '}
                      <span style={{ color: 'var(--text-primary)' }}>{r.target}</span>
                      {r.confidence && (
                        <span style={{ color: 'var(--text-dim)', marginLeft: 4 }}>
                          ({r.confidence.toFixed(2)})
                        </span>
                      )}
                      {/* per-relationship page+source */}
                      {r.pages?.length > 0 && (
                        <span style={{ marginLeft: 6 }}>
                          {r.pages.map(p => (
                            <span key={p} className="page-badge">p.{p}</span>
                          ))}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </>
            )}

            <button className="btn btn-secondary btn-sm"
              style={{ marginTop: 12, width: '100%' }}
              onClick={() => { setSelected(null); setNodeDetail(null) }}>
              ✕ Close
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
