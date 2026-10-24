"""
enrichment/primekg_matcher.py
Stage 2 — Match Neo4j entities extracted in Stage 1 to PrimeKG canonical IDs.

Matching strategy (3-tier):
  1. Exact name match (case-insensitive)
  2. Alias match  (check entity's aliases list)
  3. Fuzzy match  (rapidfuzz ratio >= threshold)

PrimeKG nodes CSV expected columns: node_index, node_id, node_type, node_name, node_source
Download: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
"""

from __future__ import annotations
import csv
import os
from typing import List, Dict, Optional
from rapidfuzz import fuzz, process as rfprocess

FUZZY_THRESHOLD = int(os.getenv("PRIMEKG_FUZZY_THRESHOLD", "88"))

# PrimeKG node_type → our schema label
# Molecular types are now included — previously set to None (skipped)
PRIMEKG_TYPE_MAP = {
    "disease":             "Disease",
    "drug":                "Drug",
    "effect/phenotype":    "SideEffect",
    "phenotype":           "Symptom",
    "symptom":             "Symptom",
    "anatomy":             "Organ",
    # Molecular (was None — now properly supported)
    "gene/protein":        "Gene",
    "protein":             "Protein",
    "gene":                "Gene",
    "pathway":             "Pathway",           # was "Treatment" — incorrect
    "biological_process":  "BiologicalProcess",
    "molecular_function":  "MolecularFunction",
    "cellular_component":  "Organ",
    "exposure":            "SideEffect",
}


class PrimeKGMatcher:
    """
    Loads PrimeKG node index and matches Neo4j entity names to PrimeKG IDs.
    """

    def __init__(self, nodes_csv: str):
        """
        Args:
            nodes_csv: Path to PrimeKG nodes CSV file (node_index, node_id, node_type, node_name, ...)
        """
        self.nodes_csv = nodes_csv
        # {lower_name → {"primekg_id": str, "type": str, "name": str}}
        self._index: Dict[str, Dict] = {}
        self._load_index()

    def _load_index(self):
        print(f"[primekg_matcher] Loading node index from {self.nodes_csv} …")
        count = 0
        with open(self.nodes_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_type = row.get("node_type", "").lower().strip()
                schema_label = PRIMEKG_TYPE_MAP.get(raw_type)
                if schema_label is None:
                    continue  # skip gene/protein etc.

                name = row.get("node_name", "").strip()
                nid  = row.get("node_id",   row.get("node_index", ""))
                if name:
                    self._index[name.lower()] = {
                        "primekg_id":   str(nid),
                        "schema_label": schema_label,
                        "name":         name,
                    }
                    count += 1

        print(f"[primekg_matcher] Loaded {count:,} PrimeKG nodes (schema-relevant)")

    def match(self, neo4j_name: str, aliases: List[str] = None) -> Optional[Dict]:
        """
        Try to match a Neo4j entity name to a PrimeKG node.

        Returns:
            {"primekg_id": str, "name": str, "schema_label": str, "score": float, "method": str}
            or None if no match found.
        """
        # Tier 1: exact match on name
        lower = neo4j_name.lower().strip()
        if lower in self._index:
            return {**self._index[lower], "score": 1.0, "method": "exact"}

        # Tier 2: alias match
        for alias in (aliases or []):
            alias_lower = alias.lower().strip()
            if alias_lower in self._index:
                return {**self._index[alias_lower], "score": 0.95, "method": "alias"}

        # Tier 3: fuzzy match
        candidates = list(self._index.keys())
        result = rfprocess.extractOne(lower, candidates, scorer=fuzz.ratio)
        if result and result[1] >= FUZZY_THRESHOLD:
            matched_key = result[0]
            return {**self._index[matched_key], "score": result[1] / 100.0, "method": "fuzzy"}

        return None

    def match_all(self, neo4j_entities: List[Dict]) -> List[Dict]:
        """
        Match a list of Neo4j entity dicts to PrimeKG.

        Args:
            neo4j_entities: List of {"name": str, "type": str, "aliases": list}

        Returns:
            List of match results:
            {"neo4j_name": str, "primekg_id": str, "score": float, "method": str} | no primekg_id if unmatched
        """
        matched, unmatched = 0, 0
        results = []

        for entity in neo4j_entities:
            name    = entity["name"]
            aliases = entity.get("aliases", [])
            match   = self.match(name, aliases)

            if match:
                results.append({
                    "neo4j_name":   name,
                    "primekg_id":   match["primekg_id"],
                    "primekg_name": match["name"],
                    "score":        match["score"],
                    "method":       match["method"],
                    "matched":      True,
                })
                matched += 1
            else:
                results.append({
                    "neo4j_name": name,
                    "matched":    False,
                })
                unmatched += 1

        rate = matched / len(neo4j_entities) * 100 if neo4j_entities else 0
        print(f"[primekg_matcher] Match rate: {matched}/{len(neo4j_entities)} "
              f"({rate:.1f}%) — {unmatched} unmatched")
        return results


def load_neo4j_entities(neo4j_client) -> List[Dict]:
    """Fetch all Stage 1 entities from Neo4j for matching."""
    cypher = """
    MATCH (n)
    WHERE n.stage = 'base_graph'
    RETURN labels(n)[0] AS type, n.name AS name, n.aliases AS aliases
    """
    return neo4j_client.run_query(cypher)


if __name__ == "__main__":
    import sys, json
    matcher = PrimeKGMatcher(sys.argv[1])
    test = [
        {"name": "Metformin",       "type": "Drug",    "aliases": []},
        {"name": "Type 2 Diabetes", "type": "Disease", "aliases": ["T2DM"]},
        {"name": "hypertension",    "type": "Disease", "aliases": []},
    ]
    results = matcher.match_all(test)
    print(json.dumps(results, indent=2))
