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
}

// Same list as NODE_LABELS in graph/neo4j_client.py
const LABELS = Object.keys(LABEL_COLORS)

// Node sources are stored as file stems ("diabetes_guide"), documents as
// filenames ("diabetes_guide.pdf"); compare both in the same form.
const docKey = s => String(s || '').split(/[\\/]/).pop().replace(/\.pdf$/i, '').toLowerCase()

// Cytoscape refuses to build a graph with a duplicate id or an edge whose
// endpoint is missing, and then draws nothing — so clean the data first.
function buildCyElements(nodes, edges) {
  const els  = []
  const seen = new Set()
  nodes.forEach(n => {
    if (!n.name || seen.has(n.name)) return
    seen.add(n.name)
    els.push({
      data: {
        id: n.name, label: n.name, nodeType: n.label || 'Unknown',
        confidence: n.confidence, stage: n.stage,
        sources: n.sources, pages: n.pages,
        umls: n.umls_cui, mesh: n.mesh_id, rxnorm: n.rxnorm_id,
      }
    })
  })
  edges.forEach((e, i) => {
    if (seen.has(e.source) && seen.has(e.target)) els.push({
      data: {
        id: `e${i}`, source: e.source, target: e.target,
        label: e.relation, confidence: e.confidence,
        pages: e.pages, sources: e.sources,
      }
    })
  })
  return els
}

// Explore results carry provenance; subgraph results only add what is missing.
function mergeGraphs(base, extra) {
  const names = new Set(base.nodes.map(n => n.name))
  const edgeKey = e => `${e.source}|${e.relation}|${e.target}`
  const edges = new Set(base.edges.map(edgeKey))
  return {
    nodes: [...base.nodes, ...extra.nodes.filter(n => n.name && !names.has(n.name))],
    edges: [...base.edges, ...extra.edges.filter(e => !edges.has(edgeKey(e)))],
  }
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
        'background-color': '#6e7681',
        'border-width': 2,
        'border-color': '#30363d',
        'transition-property': 'background-color, border-color, width, height, opacity',
        'transition-duration': '0.15s',
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
        'target-arrow-color': '#484f58',
        'line-color': '#484f58',
        'width': 1.5,
        'arrow-scale': 0.8,
      }
    },
    ...nodeStyles,
    { selector: '.faded', style: { 'opacity': 0.12 } },
    { selector: 'node.focus', style: { 'border-color': '#e3b341', 'border-width': 4 } },
    {
      selector: 'node:selected',
      style: { 'border-color': '#ffffff', 'border-width': 3, 'width': 36, 'height': 36 }
    },
    {
      selector: 'edge:selected',
      style: { 'line-color': '#388bfd', 'target-arrow-color': '#388bfd', 'width': 2.5 }
    },
  ]
}

// Fade everything except the query's entities (and their neighbours) and/or
// the nodes that came from one document.
function applyFocus(cy, highlight, docFilter) {
  cy.elements().removeClass('faded focus')
  let keep = null
  const hits = cy.nodes().filter(n => highlight.includes(n.id()))
  if (hits.length) {
    hits.addClass('focus')
    keep = hits.closedNeighborhood()
  }
  if (docFilter) {
    const key = docKey(docFilter)
    const inDoc = cy.nodes().filter(n => (n.data('sources') || []).some(s => docKey(s) === key))
    const nodes = keep ? keep.nodes().intersection(inDoc) : inDoc
    keep = nodes.union(nodes.edgesWith(nodes))
  }
  if (keep) cy.elements().not(keep).addClass('faded')
}

// ── Main component ────────────────────────────────────────────────────────────

export function GraphExplorer({ highlightEntities = [], documents = [], onClearHighlight }) {
  const cyRef       = useRef(null)
  const cyInstance  = useRef(null)
  const layoutRef   = useRef(null)
  const requestId   = useRef(0)   // only the latest fetch may draw
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [selectedNode, setSelected]   = useState(null)
  const [labelFilter, setLabelFilter] = useState('')
  const [minConf, setMinConf]         = useState(0)
  const [nodeDetail, setNodeDetail]   = useState(null)
  const [sourceDoc, setSourceDoc]     = useState('')   // filename to focus on
  const [graph, setGraph]             = useState({ nodes: [], edges: [] })

  // Read inside cytoscape callbacks, which outlive a single render
  const focusRef = useRef({ highlight: highlightEntities, doc: sourceDoc })
  focusRef.current = { highlight: highlightEntities, doc: sourceDoc }

  const highlightKey = highlightEntities.join('\u0000')

  const fetchAndRender = useCallback(async () => {
    const id = ++requestId.current
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams({ limit: 80, min_conf: minConf })
      if (labelFilter) params.append('label', labelFilter)
      const res  = await fetch(`${API}/graph/explore?${params}`)
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
      let merged = { nodes: data.nodes || [], edges: data.edges || [] }

      // The query's entities may not be in the first 80 nodes: add them
      const wanted = focusRef.current.highlight
      if (wanted.length) {
        const sub = await fetch(`${API}/graph/subgraph?${new URLSearchParams({
          entities: wanted.join(','), depth: 1 })}`)
          .then(r => (r.ok ? r.json() : { nodes: [], edges: [] }))
          .catch(() => ({ nodes: [], edges: [] }))
        merged = mergeGraphs(merged, { nodes: sub.nodes || [], edges: sub.edges || [] })
      }
      if (id !== requestId.current) return   // a newer request is on its way
      setGraph(merged)
      renderGraph(merged.nodes, merged.edges)
    } catch (e) {
      if (id !== requestId.current) return
      console.error('[graph] fetch failed:', e)
      setError(e.message === 'Failed to fetch'
        ? 'Cannot reach the API. Is the backend running on port 8000?'
        : `Could not load the graph: ${e.message}`)
    } finally {
      if (id === requestId.current) setLoading(false)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [labelFilter, minConf, highlightKey])

  useEffect(() => { fetchAndRender() }, [fetchAndRender])

  // Re-apply the document filter without refetching
  useEffect(() => {
    if (cyInstance.current) applyFocus(cyInstance.current, highlightEntities, sourceDoc)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceDoc])

  useEffect(() => () => {
    requestId.current++
    destroyGraph()
  }, [])

  // Stop the animated layout first: its next frame would touch a destroyed graph
  function destroyGraph() {
    layoutRef.current?.stop()
    cyInstance.current?.destroy()
    layoutRef.current = null
    cyInstance.current = null
  }

  function renderGraph(nodes, edges) {
    if (!cyRef.current) return
    destroyGraph()
    setSelected(null)
    setNodeDetail(null)

    const cy = cytoscape({
      container: cyRef.current,
      elements:  buildCyElements(nodes, edges),
      style:     buildCyStyle(),
      minZoom: 0.3, maxZoom: 4,
    })
    applyFocus(cy, focusRef.current.highlight, focusRef.current.doc)
    layoutRef.current = cy.layout({ name: 'cose', animate: true, padding: 30,
                                    nodeRepulsion: 4500, gravity: 0.15 })
    layoutRef.current.run()

    cy.on('tap', 'node', async evt => {
      const node = evt.target
      setSelected({ name: node.data('label'), ...node.data() })
      setNodeDetail(null)
      try {
        const res  = await fetch(`${API}/graph/node/${encodeURIComponent(node.data('label'))}`)
        setNodeDetail(res.ok ? await res.json() : null)
      } catch { setNodeDetail(null) }
    })

    cy.on('tap', function(evt) {
      if (evt.target === cy) { setSelected(null); setNodeDetail(null) }
    })

    cyInstance.current = cy
  }

  const nodeCount = new Set(graph.nodes.map(n => n.name).filter(Boolean)).size
  const docCount  = sourceDoc
    ? graph.nodes.filter(n => (n.sources || []).some(s => docKey(s) === docKey(sourceDoc))).length
    : 0

  // Legend: the types actually on screen, most common first
  const typeCounts = graph.nodes.reduce((acc, n) => {
    if (n.name && LABEL_COLORS[n.label]) acc[n.label] = (acc[n.label] || 0) + 1
    return acc
  }, {})
  const legend = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])

  return (
    <div className="card">
      <div className="card-title">🕸️ Knowledge Graph Explorer</div>

      {/* ── Controls row ─────────────────────────────────────────────── */}
      <div className="graph-controls">
        <select className="select-input" aria-label="Entity type"
          value={labelFilter} onChange={e => setLabelFilter(e.target.value)}>
          <option value="">All entity types</option>
          {LABELS.map(l => <option key={l} value={l}>{l}</option>)}
        </select>

        <label className="toggle-row">
          Min confidence:
          <input type="range" min="0" max="1" step="0.05" value={minConf}
            onChange={e => setMinConf(parseFloat(e.target.value))}
            style={{ width: 80, accentColor: 'var(--accent-blue)' }} />
          <span style={{ fontSize: 12, fontVariantNumeric: 'tabular-nums' }}>{minConf.toFixed(2)}</span>
        </label>

        <button className="btn btn-secondary btn-sm" onClick={fetchAndRender} disabled={loading}>
          {loading ? <span className="spinner" style={{ width: 12, height: 12 }} /> : '↻'} Refresh
        </button>

        <span className="graph-count">
          {nodeCount} node{nodeCount !== 1 ? 's' : ''} · {graph.edges.length} edge{graph.edges.length !== 1 ? 's' : ''}
        </span>
      </div>

      {legend.length > 0 && (
        <div className="graph-legend">
          {legend.map(([l, n]) => (
            <button key={l} type="button"
              className={`legend-item ${labelFilter === l ? 'active' : ''}`}
              title={labelFilter === l ? 'Show all types' : `Show only ${l}`}
              onClick={() => setLabelFilter(labelFilter === l ? '' : l)}>
              <span className="legend-dot" style={{ background: LABEL_COLORS[l] }} />
              {l} <span className="legend-count">{n}</span>
            </button>
          ))}
        </div>
      )}

      {/* ── Query highlight banner ───────────────────────────────────── */}
      {highlightEntities.length > 0 && (
        <div className="graph-highlight-banner">
          <span>🔎 Highlighting {highlightEntities.length} entit{highlightEntities.length !== 1 ? 'ies' : 'y'} from your last question:</span>
          <span className="chip-row">
            {highlightEntities.slice(0, 6).map(e => <span key={e} className="chip orange">{e}</span>)}
            {highlightEntities.length > 6 && <span className="chip orange">+{highlightEntities.length - 6}</span>}
          </span>
          {onClearHighlight && (
            <button className="btn btn-secondary btn-sm" style={{ marginLeft: 'auto' }}
              onClick={onClearHighlight}>✕ Clear</button>
          )}
        </div>
      )}

      {/* ── Document Source Filter ────────────────────────────────────── */}
      <div className="graph-doc-filter">
        <label className="graph-doc-filter-label" htmlFor="graph-doc">📄 Filter by document:</label>
        <select
          id="graph-doc"
          className="select-input"
          value={sourceDoc}
          onChange={e => setSourceDoc(e.target.value)}
          style={{ minWidth: 200, maxWidth: 320 }}
        >
          <option value="">All documents</option>
          {documents.map(d => (
            <option key={d.sha256} value={d.filename}>{d.filename}</option>
          ))}
        </select>
        {sourceDoc && (
          <>
            <span className="graph-doc-filter-count">
              {docCount} node{docCount !== 1 ? 's' : ''} from this PDF
            </span>
            <button className="btn btn-secondary btn-sm" onClick={() => setSourceDoc('')}>
              ✕ Clear
            </button>
          </>
        )}
      </div>

      {/* ── Graph canvas + node detail ────────────────────────────────── */}
      <div style={{ position: 'relative' }}>
        <div id="cy-container" ref={cyRef} />

        {!loading && (error || nodeCount === 0) && (
          <div className="graph-overlay">
            {error
              ? <><div style={{ fontSize: 28 }}>⚠️</div><div>{error}</div></>
              : <><div style={{ fontSize: 28 }}>🕸️</div>
                  <div>{labelFilter || minConf > 0
                    ? 'No entities match these filters.'
                    : 'The graph is empty. Upload a PDF to build it.'}</div></>}
          </div>
        )}

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

            {selectedNode.sources?.length > 0 && (
              <div style={{ marginTop: 10 }}>
                <div className="detail-heading">Source documents</div>
                {selectedNode.sources.map((s, i) => (
                  <div key={i} className="detail-source" title={s}>📄 {s}</div>
                ))}
              </div>
            )}

            {nodeDetail?.relationships?.length > 0 && (
              <>
                <div className="detail-heading" style={{
                  borderTop: '1px solid var(--border)', marginTop: 12, paddingTop: 12,
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
                      {r.confidence != null && (
                        <span style={{ color: 'var(--text-dim)', marginLeft: 4 }}>
                          ({r.confidence.toFixed(2)})
                        </span>
                      )}
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
