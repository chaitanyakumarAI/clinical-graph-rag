"""
ingestion/deduplicator.py
Stage 1 — Deduplicate and normalize extracted entities and relations.

Phase 1 enhancements:
  - Aggregate entity-level confidence (weighted average across duplicates)
  - Propagate pages and chunk_ids through merges
  - Propagate page_numbers and paragraph_ids into deduplicated relations

Strategy:
  1. Normalize all names (title-case, strip punctuation, expand common abbreviations).
  2. Group aliases: if entity A's alias matches entity B's name → merge.
  3. Fuzzy-match remaining names (rapidfuzz ratio >= 90) → merge near-duplicates.
  4. Remap relation endpoints to deduplicated canonical names.
"""

from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Tuple
from rapidfuzz import fuzz, process as rfprocess

from ingestion.entity_extractor import ExtractedEntity, ExtractedRelation, ExtractionResult

# ── Abbreviation table ────────────────────────────────────────────────────────
ABBREV_MAP: Dict[str, str] = {
    "t2dm":   "Type 2 Diabetes",
    "dm2":    "Type 2 Diabetes",
    "t1dm":   "Type 1 Diabetes",
    "htn":    "Hypertension",
    "bp":     "Blood Pressure",
    "cad":    "Coronary Artery Disease",
    "ckd":    "Chronic Kidney Disease",
    "chf":    "Heart Failure",
    "hf":     "Heart Failure",
    "gi":     "Gastrointestinal",
    "nsaid":  "Non-Steroidal Anti-Inflammatory Drug",
    "ace":    "ACE Inhibitor",
    "arb":    "Angiotensin Receptor Blocker",
    "copd":   "Chronic Obstructive Pulmonary Disease",
    "mi":     "Myocardial Infarction",
    "af":     "Atrial Fibrillation",
    "dvt":    "Deep Vein Thrombosis",
    "ampk":   "AMPK",
    "egfr":   "EGFR",
    "vegf":   "VEGF",
    "tnf":    "Tumor Necrosis Factor",
    "il-6":   "Interleukin-6",
    "il6":    "Interleukin-6",
}

FUZZY_THRESHOLD = 90  # rapidfuzz ratio threshold for merging


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Normalize an entity name to a canonical form."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKC", name)
    name = name.strip().strip(".,;:()[]")
    name = re.sub(r"\s+", " ", name)
    lower = name.lower()
    if lower in ABBREV_MAP:
        return ABBREV_MAP[lower]
    words = name.split()
    if len(words) == 1 and name.isupper() and len(name) <= 6:
        return name  # keep short acronyms: AMPK, EGFR, etc.
    return " ".join(w.capitalize() for w in words)


# ── Deduplicator ──────────────────────────────────────────────────────────────

class Deduplicator:
    """
    Deduplicates entities across multiple ExtractionResult objects.

    Phase 1: aggregates confidence (weighted average), pages, and chunk_ids
    across all merged duplicates so every canonical entity has full provenance.
    """

    def __init__(self, fuzzy_threshold: int = FUZZY_THRESHOLD):
        self.fuzzy_threshold = fuzzy_threshold
        # canonical_name → {type, name, aliases, sources, confidence, pages,
        #                    chunk_ids, observation_count}
        self._entity_registry: Dict[str, Dict] = {}
        # alias_lower → canonical_name
        self._alias_map: Dict[str, str] = {}

    def _register_entity(
        self,
        entity: ExtractedEntity,
        chunk_id: str,
        source: str,
        page: int = 0,
        paragraph_index: int = 0,
    ):
        """Add an entity to the registry, merging with existing if duplicate."""
        canonical_name = normalize_name(entity.name)
        if not canonical_name:
            return

        # ── Tier 1: exact alias match ─────────────────────────────────────
        if canonical_name.lower() in self._alias_map:
            canonical = self._alias_map[canonical_name.lower()]
            reg = self._entity_registry[canonical]
            reg["aliases"].update([normalize_name(a) for a in entity.aliases])
            reg["sources"].add(source)
            reg["chunk_ids"].add(chunk_id)
            reg["pages"].add(page)
            reg["paragraph_ids"].add(paragraph_index)
            # Running weighted average of confidence
            n = reg["observation_count"]
            reg["confidence"] = (reg["confidence"] * n + entity.confidence) / (n + 1)
            reg["observation_count"] += 1
            return

        # ── Tier 2: fuzzy match against existing canonicals ───────────────
        existing = list(self._entity_registry.keys())
        if existing:
            result = rfprocess.extractOne(
                canonical_name, existing, scorer=fuzz.ratio
            )
            if result:
                match, score, _ = result
                if score >= self.fuzzy_threshold:
                    self._alias_map[canonical_name.lower()] = match
                    reg = self._entity_registry[match]
                    reg["aliases"].add(canonical_name)
                    reg["aliases"].update([normalize_name(a) for a in entity.aliases])
                    reg["sources"].add(source)
                    reg["chunk_ids"].add(chunk_id)
                    reg["pages"].add(page)
                    reg["paragraph_ids"].add(paragraph_index)
                    n = reg["observation_count"]
                    reg["confidence"] = (reg["confidence"] * n + entity.confidence) / (n + 1)
                    reg["observation_count"] += 1
                    return

        # ── New entity ────────────────────────────────────────────────────
        aliases = {normalize_name(a) for a in entity.aliases if a}
        self._entity_registry[canonical_name] = {
            "type":             entity.type,
            "name":             canonical_name,
            "aliases":          aliases,
            "sources":          {source},
            "chunk_ids":        {chunk_id},
            "pages":            {page},
            "paragraph_ids":    {paragraph_index},
            "confidence":       entity.confidence,
            "observation_count": 1,
        }
        self._alias_map[canonical_name.lower()] = canonical_name
        for alias in aliases:
            self._alias_map[alias.lower()] = canonical_name

    def _resolve_name(self, raw_name: str) -> str:
        """Resolve any name to its canonical form."""
        normalized = normalize_name(raw_name)
        return self._alias_map.get(normalized.lower(), normalized)

    def fit(self, results: List[ExtractionResult]) -> "Deduplicator":
        """Process all extraction results and build the registry."""
        for result in results:
            for entity in result.entities:
                self._register_entity(
                    entity,
                    chunk_id        = result.chunk_id,
                    source          = result.source,
                    page            = result.page,
                    paragraph_index = result.paragraph_index,
                )
        return self

    def get_entities(self) -> List[Dict]:
        """Return deduplicated entities as plain dicts ready for Neo4j loading."""
        out = []
        for name, data in self._entity_registry.items():
            out.append({
                "type":       data["type"],
                "name":       name,
                "aliases":    sorted(data["aliases"] - {name}),
                "sources":    sorted(data["sources"]),
                "chunk_ids":  sorted(data["chunk_ids"]),
                "pages":      sorted(data["pages"]),
                "confidence": round(data["confidence"], 4),
            })
        return out

    def get_relations(self, results: List[ExtractionResult]) -> List[Dict]:
        """Return deduplicated relations with canonical endpoint names and full provenance."""
        seen: Dict[Tuple, Dict] = {}

        for result in results:
            for rel in result.relations:
                subj = self._resolve_name(rel.subject)
                obj  = self._resolve_name(rel.object)
                key  = (subj, rel.predicate, obj)

                if key not in seen:
                    seen[key] = {
                        "subject_type":  rel.subject_type,
                        "subject":       subj,
                        "predicate":     rel.predicate,
                        "object_type":   rel.object_type,
                        "object":        obj,
                        "confidence":    rel.confidence,
                        "chunk_ids":     [result.chunk_id],
                        "sources":       [result.source],
                        "page_numbers":  [result.page],
                        "paragraph_ids": [result.paragraph_index],
                    }
                else:
                    entry = seen[key]
                    # Accumulate confidence (running average)
                    n = len(entry["chunk_ids"])
                    entry["confidence"] = round(
                        (entry["confidence"] * n + rel.confidence) / (n + 1), 4
                    )
                    if result.chunk_id not in entry["chunk_ids"]:
                        entry["chunk_ids"].append(result.chunk_id)
                    if result.source not in entry["sources"]:
                        entry["sources"].append(result.source)
                    if result.page not in entry["page_numbers"]:
                        entry["page_numbers"].append(result.page)
                    if result.paragraph_index not in entry["paragraph_ids"]:
                        entry["paragraph_ids"].append(result.paragraph_index)

        return list(seen.values())


def deduplicate(results: List[ExtractionResult]) -> Tuple[List[Dict], List[Dict]]:
    """
    Convenience function: deduplicate a list of ExtractionResults.

    Returns:
        (entities, relations) — both as plain dicts ready for Neo4j loading
    """
    deduplicator = Deduplicator()
    deduplicator.fit(results)
    entities  = deduplicator.get_entities()
    relations = deduplicator.get_relations(results)

    print(f"[deduplicator] {sum(len(r.entities) for r in results)} raw entities "
          f"→ {len(entities)} canonical entities")
    print(f"[deduplicator] {sum(len(r.relations) for r in results)} raw relations "
          f"→ {len(relations)} canonical relations")
    return entities, relations


if __name__ == "__main__":

    r1 = ExtractionResult(
        chunk_id="c1", source="doc1", page=1, paragraph_index=0,
        entities=[
            ExtractedEntity(type="Drug",    name="metformin", aliases=["Metformin HCl"], confidence=0.99),
            ExtractedEntity(type="Disease", name="T2DM",      aliases=[],                confidence=0.95),
            ExtractedEntity(type="Protein", name="AMPK",      aliases=["AMP-activated protein kinase"], confidence=0.92),
        ],
        relations=[
            ExtractedRelation(subject_type="Drug",    subject="metformin", predicate="TREATS",
                              object_type="Disease",  object="T2DM",       confidence=0.99),
            ExtractedRelation(subject_type="Drug",    subject="metformin", predicate="ACTIVATES",
                              object_type="Protein",  object="AMPK",       confidence=0.95),
        ]
    )
    r2 = ExtractionResult(
        chunk_id="c2", source="doc1", page=2, paragraph_index=1,
        entities=[
            ExtractedEntity(type="Drug",    name="Metformin",     aliases=[],             confidence=0.97),
            ExtractedEntity(type="Disease", name="Type 2 Diabetes", aliases=["T2DM"],     confidence=0.99),
        ],
        relations=[
            ExtractedRelation(subject_type="Drug",    subject="Metformin", predicate="TREATS",
                              object_type="Disease",  object="Type 2 Diabetes", confidence=0.95),
        ]
    )
    entities, relations = deduplicate([r1, r2])
    import json
    print(json.dumps(entities, indent=2))
    print(json.dumps(relations, indent=2))
