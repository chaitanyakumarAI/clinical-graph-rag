import { useEffect, useRef, useState, useCallback } from 'react'
import cytoscape from 'cytoscape'
import { Icon } from './Icon'
import { ENTITY_TYPES, ENTITY_LABELS, typeVar, resolveColors, humanType } from './entityTypes'

// API base — relative so Vite proxy (dev) and FastAPI static (prod) both work
const API = ''

// Node sources are stored as file stems ("diabetes_guide"), documents as
// filenames ("diabetes_guide.pdf"); compare both in the same form.
const docKey = s => String(s || '').split(/[\\/]/).pop().replace(/\.pdf$/i, '').toLowerCase()

// Cytoscape refuses to build a graph with a duplicate id or an edge whose
// endpoint is missing, and then draws nothing — so clean the data first.
function buildCyElements(nodes, edges) {
  const seen = new Set()
  const keptNodes = []
  nodes.forEach(n => {
    if (!n.name || seen.has(n.name)) return
    seen.add(n.name)
    keptNodes.push(n)
  })
  const keptEdges = edges.filter(e => seen.has(e.source) && seen.has(e.target))
  const degree = {}
  keptEdges.forEach(e => {
    degree[e.source] = (degree[e.source] || 0) + 1
    degree[e.target] = (degree[e.target] || 0) + 1
  })
  return [
    ...keptNodes.map(n => ({
      data: {
        id: n.name, label: n.name, nodeType: n.label || 'Unknown',
        size: 20 + Math.min(degree[n.name] || 0, 8) * 3,
        confidence: n.confidence, stage: n.stage,
        sources: n.sources, pages: n.pages,
        umls: n.umls_cui, mesh: n.mesh_id, rxnorm: n.rxnorm_id,
      }
    })),
    ...keptEdges.map((e, i) => ({
      data: {
        id: `e${i}`, source: e.source, target: e.target,
        label: e.relation, confidence: e.confidence,
        pages: e.pages, sources: e.sources,
      }
    })),
  ]
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

function buildCyStyle(c) {
  const typeStyles = ENTITY_LABELS.map(t => ({
    selector: `node[nodeType="${t}"]`,
    style: { 'background-color': c.series[t], shape: ENTITY_TYPES[t].shape },
  }))
  return [
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        width: 'data(size)', height: 'data(size)',
        'background-color': c.other,
        // A ring in the canvas colour keeps overlapping nodes apart
        'border-width': 2, 'border-color': c.canvas,
        'font-family': 'Inter, system-ui, sans-serif',
        'font-size': 11, 'font-weight': 500,
        color: c.text2,
        'text-valign': 'bottom', 'text-margin-y': 5,
        'text-wrap': 'ellipsis', 'text-max-width': 120,
        'text-background-color': c.canvas, 'text-background-opacity': 0.85,
        'text-background-padding': 2, 'text-background-shape': 'roundrectangle',
        'min-zoomed-font-size': 7,
        'transition-property': 'opacity', 'transition-duration': 150,
      },
    },
    ...typeStyles,
    {
      selector: 'edge',
      style: {
        width: 1.25,
        'curve-style': 'bezier',
        'line-color': c.edge,
        'target-arrow-shape': 'triangle', 'target-arrow-color': c.edge, 'arrow-scale': 0.75,
        'font-family': 'Inter, system-ui, sans-serif',
        'font-size': 9.5, color: c.muted,
        'text-rotation': 'autorotate',
        'text-background-color': c.canvas, 'text-background-opacity': 0.9,
        'text-background-padding': 2,
        'transition-property': 'opacity, line-color', 'transition-duration': 150,
      },
    },
    // Relation names only near the node under the pointer or selected
    { selector: 'edge.near', style: { label: 'data(label)', 'line-color': c.text2, 'target-arrow-color': c.text2, width: 1.75 } },
    { selector: 'node.near', style: { color: c.text, 'font-weight': 600 } },
    { selector: '.faded', style: { opacity: 0.12 } },
    {
      selector: 'node.focus',
      style: {
        'border-width': 3, 'border-color': c.highlight,
        'underlay-color': c.accent, 'underlay-opacity': 0.16, 'underlay-padding': 7,
        'underlay-shape': 'ellipse',
        color: c.text, 'font-weight': 600,
      },
    },
    { selector: 'node:selected', style: { 'border-width': 3, 'border-color': c.highlight, color: c.text, 'font-weight': 650 } },
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

// Lay out the connected part with a force layout, then park the nodes with no
// edges in a tidy grid underneath instead of a long stray column.
function runLayout(cy) {
  const isolated  = cy.nodes().filter(n => n.degree() === 0)
  const connected = cy.elements().not(isolated)
  // Start from a box with the canvas's shape, so the result fills it
  const W = Math.max(cy.width(), 600)
  const H = Math.max(cy.height() * (isolated.length ? 0.7 : 1), 360)
  if (connected.length) {
    connected.layout({
      name: 'cose', animate: false, randomize: true, fit: false,
      boundingBox: { x1: 0, y1: 0, w: W, h: H },
      idealEdgeLength: 110, edgeElasticity: 100, nodeRepulsion: 40000, nodeOverlap: 30,
      componentSpacing: 100, gravity: 0.15, numIter: 3000,
    }).run()
  }
  if (isolated.length) {
    const cell = 130
    const box  = connected.length ? connected.boundingBox() : { x1: 0, y2: -90, w: W }
    const cols = Math.max(6, Math.floor(Math.max(box.w, W * 0.8) / cell))
    isolated.layout({
      name: 'grid', fit: false, avoidOverlap: true, cols,
      boundingBox: { x1: box.x1 + (box.w - cols * cell) / 2, y1: box.y2 + 90, w: cols * cell,
                     h: Math.ceil(isolated.length / cols) * 60 },
    }).run()
  }
  cy.fit(undefined, 40)
}

// Small SVG key matching the node's shape and colour
function ShapeSwatch({ type, size = 12 }) {
  const shape = ENTITY_TYPES[type]?.shape || 'ellipse'
  const fill  = typeVar(type)
  const r = size / 2
  const poly = (n, rot = -90, k = 1) => Array.from({ length: n }, (_, i) => {
    const a = ((rot + (360 / n) * i) * Math.PI) / 180
    return `${r + r * k * Math.cos(a)},${r + r * k * Math.sin(a)}`
  }).join(' ')
  const star = Array.from({ length: 10 }, (_, i) => {
    const a = ((-90 + 36 * i) * Math.PI) / 180, k = i % 2 ? 0.45 : 1
    return `${r + r * k * Math.cos(a)},${r + r * k * Math.sin(a)}`
  }).join(' ')
  const s = size
  const el = {
    'ellipse':         <circle cx={r} cy={r} r={r} />,
    'triangle':        <polygon points={poly(3)} />,
    'round-triangle':  <polygon points={poly(3)} />,
    'diamond':         <polygon points={poly(4)} />,
    'round-diamond':   <polygon points={poly(4)} />,
    'pentagon':        <polygon points={poly(5)} />,
    'hexagon':         <polygon points={poly(6, 0)} />,
    'round-hexagon':   <polygon points={poly(6, 0)} />,
    'octagon':         <polygon points={poly(8, 22.5)} />,
    'star':            <polygon points={star} />,
    'round-rectangle': <rect width={s} height={s} rx={s * 0.25} />,
    'barrel':          <rect x={0} y={s * 0.1} width={s} height={s * 0.8} rx={s * 0.4} />,
    'tag':             <polygon points={`0,0 ${s * 0.65},0 ${s},${r} ${s * 0.65},${s} 0,${s}`} />,
    'vee':             <polygon points={`0,0 ${r},${s * 0.4} ${s},0 ${r},${s}`} />,
    'rhomboid':        <polygon points={`${s * 0.3},0 ${s},0 ${s * 0.7},${s} 0,${s}`} />,
  }[shape]
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true"
      style={{ fill, flexShrink: 0, overflow: 'visible' }}>{el}</svg>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function GraphExplorer({ highlightEntities = [], documents = [], onClearHighlight, theme }) {
  const cyRef       = useRef(null)
  const cyInstance  = useRef(null)
  const requestId   = useRef(0)   // only the latest fetch may draw
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [selectedNode, setSelected]   = useState(null)
  const [labelFilter, setLabelFilter] = useState('')
  const [minConf, setMinConf]         = useState(0)
  const [nodeDetail, setNodeDetail]   = useState(null)
  const [sourceDoc, setSourceDoc]     = useState('')   // filename to focus on
  const [graph, setGraph]             = useState({ nodes: [], edges: [] })
  // Nodes with no relation on screen crowd the picture: hidden unless asked for
  const [showIsolated, setShowIsolated] = useState(false)
  const [hiddenCount, setHiddenCount]   = useState(0)

  // Read inside cytoscape callbacks, which outlive a single render
  const focusRef = useRef({ highlight: highlightEntities, doc: sourceDoc, showIsolated })
  focusRef.current = { highlight: highlightEntities, doc: sourceDoc, showIsolated }

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

  // Redraw (no refetch) when unconnected nodes are shown or hidden
  const firstToggle = useRef(true)
  useEffect(() => {
    if (firstToggle.current) { firstToggle.current = false; return }
    renderGraph(graph.nodes, graph.edges)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showIsolated])

  // Re-apply the document filter without refetching
  useEffect(() => {
    if (cyInstance.current) applyFocus(cyInstance.current, highlightEntities, sourceDoc)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceDoc])

  // Repaint in the new theme's colours. Wait a frame: App sets data-theme in
  // its own effect, which runs after this one.
  useEffect(() => {
    const frame = requestAnimationFrame(() =>
      cyInstance.current?.style(buildCyStyle(resolveColors())))
    return () => cancelAnimationFrame(frame)
  }, [theme])

  // Keep the drawing in sync with the canvas size (window resize, layout shifts)
  useEffect(() => {
    const el = cyRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const cy = cyInstance.current
      if (!cy) return
      cy.resize()
      if (!cy.nodes(':selected').length) cy.fit(undefined, 40)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => () => {
    requestId.current++
    destroyGraph()
  }, [])

  function destroyGraph() {
    cyInstance.current?.destroy()
    cyInstance.current = null
  }

  function closeDetail() {
    cyInstance.current?.elements().removeClass('near').unselect()
    setSelected(null)
    setNodeDetail(null)
  }

  function renderGraph(nodes, edges) {
    if (!cyRef.current) return
    destroyGraph()
    setSelected(null)
    setNodeDetail(null)

    const cy = cytoscape({
      container: cyRef.current,
      elements:  buildCyElements(nodes, edges),
      style:     buildCyStyle(resolveColors()),
      minZoom: 0.2, maxZoom: 3,
      wheelSensitivity: 0.25,
    })
    let hidden = 0
    if (!focusRef.current.showIsolated) {
      const lone = cy.nodes().filter(n => n.degree() === 0 && !focusRef.current.highlight.includes(n.id()))
      hidden = lone.length
      // Keep at least something on screen
      if (hidden < cy.nodes().length) lone.remove()
      else hidden = 0
    }
    setHiddenCount(hidden)
    runLayout(cy)
    applyFocus(cy, focusRef.current.highlight, focusRef.current.doc)

    const markNear = node => {
      cy.elements().removeClass('near')
      node.connectedEdges().addClass('near')
      node.neighborhood('node').addClass('near')
    }
    const setCursor = value => { if (cyRef.current) cyRef.current.style.cursor = value }
    cy.on('mouseover', 'node', evt => {
      setCursor('pointer')
      markNear(evt.target)
    })
    cy.on('mouseout', 'node', () => {
      setCursor('')
      const sel = cy.nodes(':selected')
      if (sel.length) markNear(sel)
      else cy.elements().removeClass('near')
    })

    cy.on('tap', 'node', async evt => {
      const node = evt.target
      markNear(node)
      setSelected({ name: node.data('label'), ...node.data() })
      setNodeDetail(null)
      try {
        const res  = await fetch(`${API}/graph/node/${encodeURIComponent(node.data('label'))}`)
        setNodeDetail(res.ok ? await res.json() : null)
      } catch { setNodeDetail(null) }
    })

    cy.on('tap', evt => {
      if (evt.target === cy) {
        cy.elements().removeClass('near')
        setSelected(null); setNodeDetail(null)
      }
    })

    cyInstance.current = cy
  }

  const zoomBy = factor => {
    const cy = cyInstance.current
    if (!cy) return
    cy.zoom({ level: cy.zoom() * factor, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } })
  }

  const nodeCount = new Set(graph.nodes.map(n => n.name).filter(Boolean)).size
  const docCount  = sourceDoc
    ? graph.nodes.filter(n => (n.sources || []).some(s => docKey(s) === docKey(sourceDoc))).length
    : 0

  // Legend: the types actually on screen, most common first
  const typeCounts = graph.nodes.reduce((acc, n) => {
    if (n.name) acc[n.label || 'Unknown'] = (acc[n.label || 'Unknown'] || 0) + 1
    return acc
  }, {})
  const legend = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])

  // Relationships to named entities (evidence links have no target)
  const rels = (nodeDetail?.relationships || []).filter(r => r.target)

  return (
    <div className="page wide">
      <div className="page-header">
        <div>
          <h1 className="page-title">Knowledge graph</h1>
          <p className="page-desc">
            Entities and relations extracted from your documents. Hover a node to see its
            relations, click it for its sources.
          </p>
        </div>
      </div>

      <div className="card graph-stage">
        {/* Filters */}
        <div className="graph-bar">
          <select className="select" aria-label="Entity type"
            value={labelFilter} onChange={e => setLabelFilter(e.target.value)}>
            <option value="">All entity types</option>
            {ENTITY_LABELS.map(l => <option key={l} value={l}>{humanType(l)}</option>)}
          </select>
          <select className="select" aria-label="Source document"
            value={sourceDoc} onChange={e => setSourceDoc(e.target.value)}
            style={{ maxWidth: 260 }}>
            <option value="">All documents</option>
            {documents.map(d => <option key={d.sha256} value={d.filename}>{d.filename.replace(/\.pdf$/i, '')}</option>)}
          </select>
          <label className="conf-control">
            Min. confidence
            <input type="range" min="0" max="1" step="0.05" value={minConf}
              onChange={e => setMinConf(parseFloat(e.target.value))} />
            <span className="num" style={{ minWidth: 28 }}>{minConf.toFixed(2)}</span>
          </label>
          <button type="button" className="btn btn-ghost btn-sm" style={{ marginLeft: 'auto' }}
            onClick={fetchAndRender} disabled={loading}>
            {loading ? <span className="spinner" style={{ width: 14, height: 14 }} /> : <Icon name="refresh" size={15} />}
            Reload
          </button>
        </div>

        <div className="graph-canvas">
        <div id="cy-container" ref={cyRef} />

        <div className="floating zoom-controls">
          <button type="button" className="btn btn-ghost btn-sm btn-icon" onClick={() => zoomBy(1.25)} aria-label="Zoom in" title="Zoom in"><Icon name="plus" size={16} /></button>
          <button type="button" className="btn btn-ghost btn-sm btn-icon" onClick={() => zoomBy(0.8)} aria-label="Zoom out" title="Zoom out"><Icon name="minus" size={16} /></button>
          <button type="button" className="btn btn-ghost btn-sm btn-icon" onClick={() => cyInstance.current?.fit(undefined, 40)} aria-label="Fit to screen" title="Fit to screen"><Icon name="fit" size={16} /></button>
        </div>

        {/* Query highlight */}
        {highlightEntities.length > 0 && (
          <div className="highlight-pill">
            <span className="ring" aria-hidden="true" />
            <span className="label">
              Showing {highlightEntities.length} entit{highlightEntities.length !== 1 ? 'ies' : 'y'} from your last answer
            </span>
            {onClearHighlight && (
              <button type="button" className="btn btn-ghost btn-sm" onClick={onClearHighlight}>
                <Icon name="x" size={14} />Clear
              </button>
            )}
          </div>
        )}

        {!loading && (error || nodeCount === 0) && (
          <div className="graph-overlay">
            <div className="empty">
              <div className="empty-icon"><Icon name={error ? 'alert' : 'graph'} size={22} /></div>
              <div className="empty-title">{error ? 'Graph unavailable' : 'Nothing to show'}</div>
              <div>{error || (labelFilter || minConf > 0
                ? 'No entities match these filters.'
                : 'The graph is empty. Upload a PDF to build it.')}</div>
            </div>
          </div>
        )}

        {selectedNode && (
          <aside className="drawer" aria-label={`Details for ${selectedNode.name}`}>
            <div className="drawer-head">
              <div style={{ minWidth: 0 }}>
                <div className="drawer-title">{selectedNode.name}</div>
                <div className="drawer-type">
                  <ShapeSwatch type={selectedNode.nodeType} />{humanType(selectedNode.nodeType)}
                  {selectedNode.confidence != null && (
                    <span className="muted num">· confidence {Number(selectedNode.confidence).toFixed(2)}</span>
                  )}
                </div>
              </div>
              <button type="button" className="btn btn-ghost btn-sm btn-icon" style={{ marginLeft: 'auto' }}
                onClick={closeDetail} aria-label="Close details"><Icon name="x" size={16} /></button>
            </div>
            <div className="drawer-body">
              {(selectedNode.umls || selectedNode.mesh || selectedNode.rxnorm) && (
                <dl className="kv" style={{ gridTemplateColumns: '90px 1fr' }}>
                  {selectedNode.umls   && <><dt>UMLS</dt><dd className="mono">{selectedNode.umls}</dd></>}
                  {selectedNode.mesh   && <><dt>MeSH</dt><dd className="mono">{selectedNode.mesh}</dd></>}
                  {selectedNode.rxnorm && <><dt>RxNorm</dt><dd className="mono">{selectedNode.rxnorm}</dd></>}
                </dl>
              )}

              {selectedNode.sources?.length > 0 && (
                <div>
                  <div className="section-label">Found in</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {selectedNode.sources.map((s, i) => (
                      <div key={i} className="doc-pill" title={s}>
                        <Icon name="file" size={14} /><span>{String(s).split(/[\\/]/).pop()}</span>
                      </div>
                    ))}
                  </div>
                  {selectedNode.pages?.length > 0 && (
                    <div className="muted" style={{ fontSize: 12.5, marginTop: 6 }}>
                      Page{selectedNode.pages.length !== 1 ? 's' : ''} {selectedNode.pages.join(', ')}
                    </div>
                  )}
                </div>
              )}

              <div>
                <div className="section-label">Relations {nodeDetail && <span className="num">({rels.length})</span>}</div>
                {!nodeDetail && <div className="skeleton" style={{ width: '70%' }} />}
                {nodeDetail && rels.length === 0 && <div className="muted" style={{ fontSize: 13 }}>No relations to other entities.</div>}
                <div className="rel-list">
                  {rels.slice(0, 12).map((r, i) => (
                    <div key={i} className="rel">
                      <div className="rel-type">{r.relation}</div>
                      <div>{r.target}</div>
                      {(r.confidence != null || r.pages?.length > 0) && (
                        <div className="rel-meta num">
                          {r.confidence != null && <>confidence {r.confidence.toFixed(2)}</>}
                          {r.pages?.length > 0 && <> · p. {r.pages.join(', ')}</>}
                        </div>
                      )}
                    </div>
                  ))}
                  {rels.length > 12 && <div className="muted" style={{ fontSize: 12.5 }}>+{rels.length - 12} more</div>}
                </div>
              </div>
            </div>
          </aside>
        )}
        </div>

        {/* Legend (click a type to show only that type) and counts */}
        <div className="graph-foot">
          <div className="legend-items" role="group" aria-label="Entity types">
            {legend.map(([l, n]) => (
              <button key={l} type="button"
                className={`legend-item ${labelFilter === l ? 'active' : ''}`}
                aria-pressed={labelFilter === l}
                title={labelFilter === l ? 'Show all types' : `Show only ${humanType(l)}`}
                onClick={() => setLabelFilter(labelFilter === l ? '' : l)}>
                <ShapeSwatch type={l} />
                {humanType(l)} <span className="count">{n}</span>
              </button>
            ))}
          </div>
          <div className="graph-stats" aria-live="polite">
            {nodeCount - hiddenCount} node{nodeCount - hiddenCount !== 1 ? 's' : ''} · {graph.edges.length} relation{graph.edges.length !== 1 ? 's' : ''}
            {sourceDoc && <> · <strong>{docCount}</strong> in this document</>}
            {(hiddenCount > 0 || showIsolated) && (
              <button type="button" className="btn btn-ghost btn-sm" style={{ marginLeft: 6 }}
                onClick={() => setShowIsolated(s => !s)}>
                {showIsolated ? 'Hide unconnected' : `Show ${hiddenCount} unconnected`}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
