"""
enrichment/primekg_edge_filter.py
Stage 2 — Filter and map PrimeKG edges to our schema predicates.

Key fixes vs. the previous version:
  - drug_protein   → TARGETS   (was wrongly AFFECTS_ORGAN — protein ≠ organ)
  - disease_protein→ BIOMARKER_FOR (not AFFECTS_ORGAN)
  - gene/protein   → Gene      (was skipped / mapped to Organ)
  - pathway        → Pathway   (was Treatment)
  - biological_process → BiologicalProcess (was skipped)
  - molecular_function → MolecularFunction (was skipped)

Every mapped edge is additionally validated through the central ontology_validator
before being added to the output list.
"""

from __future__ import annotations
import csv
from typing import List, Dict, Set

from ingestion.ontology_validator import validate_triple

# ── PrimeKG relation → our schema predicate ───────────────────────────────────
# Separate molecular-level from organ-level relations.

RELATION_MAP: Dict[str, str] = {
    # Drug ↔ Disease
    "drug_disease":                "TREATS",
    "disease_drug":                "TREATS",
    "indication":                  "TREATS",

    # Drug ↔ Protein/Gene — molecular target (NOT organ effect)
    "drug_protein":                "TARGETS",
    "protein_drug":                "TARGETS",
    "gene_drug":                   "TARGETS",

    # Disease ↔ Protein/Gene — biomarker / driver
    "disease_protein":             "BIOMARKER_FOR",
    "protein_disease":             "BIOMARKER_FOR",

    # Protein ↔ Protein — protein–protein interaction
    "protein_protein":             "INTERACTS_WITH",

    # Drug side-effects
    "drug_effect":                 "CAUSES",
    "side_effect":                 "CAUSES",

    # Contraindication
    "contraindication":            "CONTRAINDICATED_FOR",

    # Disease ↔ Phenotype / Symptom
    "disease_phenotype_positive":  "HAS_SYMPTOM",
    "phenotype_disease":           "HAS_SYMPTOM",
    "disease_symptom":             "HAS_SYMPTOM",

    # Drug ↔ Drug interaction
    "drug_drug":                   "INTERACTS_WITH",

    # Anatomy (Organ) ↔ Disease — organ expresses / hosts disease
    "anatomy_disease_present":     "EXPRESSED_IN",   # not AFFECTS_ORGAN — direction is from gene/protein
    "anatomy_disease":             "EXPRESSED_IN",

    # Pathway ↔ Disease
    "pathway_disease":             "LEADS_TO",
    "disease_pathway":             "LEADS_TO",

    # Pathway ↔ Drug
    "pathway_drug":                "PARTICIPATES_IN",
    "drug_pathway":                "PARTICIPATES_IN",

    # Biological process / molecular function
    "biological_process_gene":     "PARTICIPATES_IN",
    "gene_biological_process":     "PARTICIPATES_IN",
    "molecular_function_gene":     "PARTICIPATES_IN",

    # Exposure ↔ Disease
    "exposure_disease":            "CAUSES",
    "disease_exposure":            "CAUSES",

    # Cellular component ↔ Gene
    "cellular_component_gene":     "EXPRESSED_IN",
}

# ── PrimeKG x_type/y_type → our schema label ─────────────────────────────────

TYPE_MAP: Dict[str, str] = {
    "disease":              "Disease",
    "drug":                 "Drug",
    "effect/phenotype":     "SideEffect",
    "phenotype":            "Symptom",
    "symptom":              "Symptom",
    "anatomy":              "Organ",
    # Molecular types — now properly included (were None / skipped before)
    "gene/protein":         "Gene",
    "protein":              "Protein",
    "gene":                 "Gene",
    "pathway":              "Pathway",          # was "Treatment" — incorrect
    "biological_process":   "BiologicalProcess",
    "molecular_function":   "MolecularFunction",
    "cellular_component":   "Organ",            # cell compartment treated as organ-level
    "exposure":             "SideEffect",
}


def filter_edges(
    edges_csv:           str,
    matched_primekg_ids: Set[str],
    id_to_neo4j:         Dict[str, str],
    confidence:          float = 0.85,
) -> List[Dict]:
    """
    Read PrimeKG edges CSV and keep only edges where BOTH endpoints are matched
    AND the mapped relation passes ontology validation.

    Args:
        edges_csv:           Path to PrimeKG edges CSV
        matched_primekg_ids: Set of PrimeKG IDs matched to Neo4j entities
        id_to_neo4j:         {primekg_id → neo4j_canonical_name}
        confidence:          Confidence score assigned to all PrimeKG edges

    Returns:
        List of filtered edge dicts ready for Neo4j upsert
    """
    filtered:       List[Dict] = []
    total = kept = skipped_endpoint = skipped_type = skipped_ontology = 0

    print(f"[edge_filter] Scanning {edges_csv} …")
    print(f"[edge_filter] Matched PrimeKG IDs: {len(matched_primekg_ids)}")

    with open(edges_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total % 500_000 == 0:
                print(f"  … {total:,} rows scanned, {kept:,} kept", end="\r")

            x_id = str(row.get("x_id", row.get("x_index", "")))
            y_id = str(row.get("y_id", row.get("y_index", "")))

            # Both endpoints must be matched to our graph
            if x_id not in matched_primekg_ids or y_id not in matched_primekg_ids:
                skipped_endpoint += 1
                continue

            relation   = row.get("relation", "").lower().strip()
            predicate  = RELATION_MAP.get(relation, "RELATED_TO")

            x_type_raw = row.get("x_type", "").lower().strip()
            y_type_raw = row.get("y_type", "").lower().strip()
            x_label    = TYPE_MAP.get(x_type_raw)
            y_label    = TYPE_MAP.get(y_type_raw)

            if not x_label or not y_label:
                skipped_type += 1
                continue

            # Ontology validation — ensures protein relations don't collapse to organ
            val = validate_triple(x_label, predicate, y_label)
            if not val.is_valid:
                # Try reversing direction for asymmetric predicates
                if val.suggested_predicate:
                    rev = validate_triple(y_label, predicate, x_label)
                    if rev.is_valid:
                        x_id, y_id         = y_id, x_id
                        x_label, y_label   = y_label, x_label
                    else:
                        skipped_ontology += 1
                        continue
                else:
                    skipped_ontology += 1
                    continue

            filtered.append({
                "subject_type":     x_label,
                "subject":          id_to_neo4j[x_id],
                "predicate":        predicate,
                "object_type":      y_label,
                "object":           id_to_neo4j[y_id],
                "confidence":       confidence,
                "primekg_relation": relation,
                "chunk_ids":        [],
                "sources":          ["primekg"],
                "stage":            "primekg_enriched",
            })
            kept += 1

    print(
        f"\n[edge_filter] Scanned {total:,} → kept {kept:,} "
        f"(dropped: {skipped_endpoint:,} no-endpoint, "
        f"{skipped_type:,} unknown-type, "
        f"{skipped_ontology:,} ontology-invalid)"
    )
    return filtered


def build_match_maps(match_results: List[Dict]) -> tuple:
    """
    Build helper data structures from primekg_matcher output.

    Returns:
        (matched_ids: Set[str], id_to_neo4j: Dict[str, str])
    """
    matched_ids: Set[str]    = set()
    id_to_neo4j: Dict[str, str] = {}
    for m in match_results:
        if m.get("matched") and "primekg_id" in m:
            pid = m["primekg_id"]
            matched_ids.add(pid)
            id_to_neo4j[pid] = m["neo4j_name"]
    return matched_ids, id_to_neo4j
