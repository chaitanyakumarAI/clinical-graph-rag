"""
ingestion/ontology_validator.py
Single source of truth for the medical knowledge graph ontology.

Defines domain (allowed subject types) and range (allowed object types)
for every predicate. Used by:
  - entity_extractor.py  — replaces inline _PRED_CONSTRAINTS
  - primekg_edge_filter.py — validates every imported PrimeKG edge

ValidationResult carries is_valid, reason, and an optional suggested_predicate
so callers can log or auto-correct direction errors rather than silently dropping.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

# ── Canonical entity types ────────────────────────────────────────────────────

ALL_ENTITY_TYPES: FrozenSet[str] = frozenset({
    "Disease", "Drug", "Symptom", "Treatment",
    "SideEffect", "PatientGroup", "Organ", "MedicalGuideline",
    "Gene", "Protein", "Pathway", "Compound",
    "CellType", "MolecularFunction", "BiologicalProcess",
})

# ── Ontology definition ────────────────────────────────────────────────────────
# (domain_types, range_types) — empty frozenset = "any type allowed"

Constraint = Tuple[FrozenSet[str], FrozenSet[str]]

ONTOLOGY: Dict[str, Constraint] = {
    # ── Clinical ──────────────────────────────────────────────────────────────
    "TREATS": (
        frozenset({"Drug", "Treatment", "Compound"}),
        frozenset({"Disease", "Symptom", "PatientGroup"}),
    ),
    "CAUSES": (
        frozenset({"Drug", "Disease", "Compound", "Treatment", "Pathway"}),
        frozenset({"Disease", "SideEffect", "Symptom"}),
    ),
    "HAS_SYMPTOM": (
        frozenset({"Disease"}),
        frozenset({"Symptom"}),
    ),
    "AFFECTS_ORGAN": (
        frozenset({"Drug", "Disease", "Compound", "Treatment"}),
        frozenset({"Organ"}),
    ),
    "CONTRAINDICATED_FOR": (
        frozenset({"Drug"}),
        frozenset({"Disease", "PatientGroup", "Symptom"}),
    ),
    "INTERACTS_WITH": (
        frozenset({"Drug", "Compound", "Protein", "Gene"}),
        frozenset({"Drug", "Compound", "Protein", "Gene"}),
    ),
    "RECOMMENDED_BY": (
        frozenset({"Treatment", "Drug", "MedicalGuideline"}),
        frozenset({"Disease", "PatientGroup", "MedicalGuideline"}),
    ),
    # ── Molecular ─────────────────────────────────────────────────────────────
    "TARGETS": (
        # Drug/Compound targets a Protein/Gene/Pathway — NOT an Organ
        frozenset({"Drug", "Compound"}),
        frozenset({"Protein", "Gene", "Pathway"}),
    ),
    "REGULATES": (
        frozenset({"Gene", "Protein"}),
        frozenset({"Gene", "Protein", "Pathway", "BiologicalProcess", "MolecularFunction"}),
    ),
    "INHIBITS": (
        frozenset({"Drug", "Compound", "Protein", "Gene"}),
        frozenset({"Protein", "Gene", "Pathway", "BiologicalProcess"}),
    ),
    "ACTIVATES": (
        frozenset({"Drug", "Compound", "Protein", "Gene"}),
        frozenset({"Protein", "Gene", "Pathway", "BiologicalProcess"}),
    ),
    "BIOMARKER_FOR": (
        # Gene/Protein is a biomarker FOR a Disease — NOT the reverse
        frozenset({"Gene", "Protein"}),
        frozenset({"Disease"}),
    ),
    "EXPRESSED_IN": (
        frozenset({"Gene", "Protein"}),
        frozenset({"CellType", "Organ"}),
    ),
    "PARTICIPATES_IN": (
        frozenset({"Gene", "Protein", "Compound", "Drug"}),
        frozenset({"Pathway", "BiologicalProcess", "MolecularFunction"}),
    ),
    "LEADS_TO": (
        frozenset({"Disease", "Pathway", "Symptom", "Compound"}),
        frozenset({"Disease", "Pathway", "Symptom"}),
    ),
    # ── Catch-all ─────────────────────────────────────────────────────────────
    "RELATED_TO": (
        frozenset(),  # unrestricted
        frozenset(),
    ),
}


@dataclass
class ValidationResult:
    is_valid:            bool
    reason:              str = ""
    suggested_predicate: Optional[str] = None  # set when direction is simply reversed


def validate_triple(
    subject_type: str,
    predicate:    str,
    object_type:  str,
) -> ValidationResult:
    """
    Validate (subject_type, predicate, object_type) against the ontology.

    Checks (in order):
      1. Predicate is known — unknown predicates pass with a note
      2. Subject type is in domain (or domain is empty = unrestricted)
      3. Object type is in range  (or range  is empty = unrestricted)
      4. Direction check — if swapping subject/object would work, report it

    Empty subject_type / object_type always pass (conservative: don't block
    triples where the LLM omitted type annotations).
    """
    if predicate not in ONTOLOGY:
        return ValidationResult(is_valid=True, reason=f"unknown predicate '{predicate}' — pass-through")

    domain, range_ = ONTOLOGY[predicate]

    subj_ok = (not subject_type) or (not domain) or (subject_type in domain)
    obj_ok  = (not object_type)  or (not range_)  or (object_type  in range_)

    if subj_ok and obj_ok:
        return ValidationResult(is_valid=True, reason="ok")

    # Check if direction is simply reversed
    rev_subj_ok = (not subject_type) or (not range_)  or (subject_type in range_)
    rev_obj_ok  = (not object_type)  or (not domain) or (object_type  in domain)
    if rev_subj_ok and rev_obj_ok:
        return ValidationResult(
            is_valid=False,
            reason=(
                f"direction reversed: {subject_type} → {predicate} → {object_type}; "
                f"swap arguments to fix"
            ),
            suggested_predicate=predicate,
        )

    parts = []
    if not subj_ok:
        parts.append(f"subject '{subject_type}' ∉ domain{sorted(domain)}")
    if not obj_ok:
        parts.append(f"object '{object_type}' ∉ range{sorted(range_)}")

    return ValidationResult(is_valid=False, reason="; ".join(parts))


def suggest_predicate(subject_type: str, object_type: str) -> Optional[str]:
    """
    Given a (subject_type, object_type) pair, return the most specific valid predicate.
    Returns None if zero or multiple predicates match (ambiguous).
    """
    matches = [
        pred for pred, (dom, rng) in ONTOLOGY.items()
        if pred != "RELATED_TO"
        and ((not dom) or subject_type in dom)
        and ((not rng) or object_type  in rng)
    ]
    return matches[0] if len(matches) == 1 else None


if __name__ == "__main__":
    tests = [
        ("Drug",    "TARGETS",      "Protein",   True),
        ("Drug",    "TARGETS",      "Organ",     False),   # was the bug
        ("Drug",    "AFFECTS_ORGAN","Organ",     True),
        ("Gene",    "BIOMARKER_FOR","Disease",   True),
        ("Disease", "BIOMARKER_FOR","Gene",      False),   # direction reversed
        ("Protein", "INTERACTS_WITH","Drug",     False),   # Protein↔Drug not allowed
        ("Drug",    "INTERACTS_WITH","Drug",     True),
        ("Drug",    "EXPRESSED_IN", "Organ",     False),   # Drug can't be expressed in
        ("Gene",    "EXPRESSED_IN", "Organ",     True),
    ]
    print(f"{'Subject':<12} {'Predicate':<20} {'Object':<12} {'Expect':<7} {'Got':<7} Reason")
    print("-" * 85)
    for subj, pred, obj, expected in tests:
        r = validate_triple(subj, pred, obj)
        icon = "✅" if r.is_valid == expected else "❌"
        print(f"{icon} {subj:<12} {pred:<20} {obj:<12} {str(expected):<7} {str(r.is_valid):<7} {r.reason}")
        if r.suggested_predicate:
            print(f"   → Suggested fix: swap arguments (keep predicate '{r.suggested_predicate}')")
