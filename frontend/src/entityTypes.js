// One fixed colour + shape per entity type, shared by the graph and the
// dashboard. The eight colours are a colour-blind-checked categorical set
// (light and dark steps); colour always follows the type, never its rank.
// Shape is a second channel, so a type never depends on colour alone.
// Types past the eighth share a neutral "other" colour and keep their shape.

export const ENTITY_TYPES = {
  Drug:              { slot: 1, shape: 'ellipse' },
  SideEffect:        { slot: 2, shape: 'triangle' },
  Symptom:           { slot: 3, shape: 'diamond' },
  Organ:             { slot: 4, shape: 'hexagon' },
  PatientGroup:      { slot: 5, shape: 'barrel' },
  Protein:           { slot: 6, shape: 'pentagon' },
  MedicalGuideline:  { slot: 7, shape: 'tag' },
  Disease:           { slot: 8, shape: 'round-rectangle' },
  CellType:          { slot: 0, shape: 'octagon' },
  Gene:              { slot: 0, shape: 'star' },
  Pathway:           { slot: 0, shape: 'vee' },
  Treatment:         { slot: 0, shape: 'round-triangle' },
  Compound:          { slot: 0, shape: 'round-diamond' },
  MolecularFunction: { slot: 0, shape: 'rhomboid' },
  BiologicalProcess: { slot: 0, shape: 'round-hexagon' },
}

// Same list as NODE_LABELS in graph/neo4j_client.py
export const ENTITY_LABELS = Object.keys(ENTITY_TYPES)

export const typeVar = type => {
  const slot = ENTITY_TYPES[type]?.slot || 0
  return slot ? `var(--series-${slot})` : 'var(--series-other)'
}

// Cytoscape cannot read CSS variables, so resolve them from the live theme
export function resolveColors(el = document.documentElement) {
  const css = getComputedStyle(el)
  const get = name => css.getPropertyValue(name).trim()
  const series = {}
  ENTITY_LABELS.forEach(t => {
    const slot = ENTITY_TYPES[t].slot
    series[t] = get(slot ? `--series-${slot}` : '--series-other')
  })
  return {
    series,
    other:     get('--series-other'),
    surface:   get('--surface'),
    canvas:    get('--canvas'),
    text:      get('--text'),
    text2:     get('--text-2'),
    muted:     get('--text-3'),
    edge:      get('--edge'),
    accent:    get('--accent'),
    highlight: get('--highlight'),
  }
}

// "SideEffect" → "Side effect"
export const humanType = t =>
  t.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/ (\w)/g, (_, c) => ' ' + c.toLowerCase())
