"""
ingestion/identifier_linker.py
Phase 2 — Map extracted entities to canonical medical identifiers.

Identifier lookup tiers (in order):
  1. RxNorm REST API    — Drug / Compound → rxnorm_id  (free, no key needed)
  2. MeSH RDF lookup   — Disease / Symptom / Gene → mesh_id (free)
  3. UMLS REST API     — all types → umls_cui  (free account at uts.nlm.nih.gov)
  4. Local alias CSV   — brand name → generic / canonical drug name lookup

Each tier is tried only when the previous one fails.
Any tier that is unavailable (no key, no network) is silently skipped.
"""

from __future__ import annotations
import os
import time
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

UMLS_API_KEY   = os.getenv("UMLS_API_KEY", "")
RXNORM_BASE    = "https://rxnav.nlm.nih.gov/REST"
UMLS_BASE      = "https://uts-ws.nlm.nih.gov/rest"
MESH_BASE      = "https://id.nlm.nih.gov/mesh/lookup"

# Simple in-process cache to avoid duplicate API hits within one run
_rxnorm_cache: Dict[str, Optional[str]] = {}
_mesh_cache:   Dict[str, Optional[str]] = {}
_umls_cache:   Dict[str, Optional[str]] = {}


# ── RxNorm ────────────────────────────────────────────────────────────────────

def _lookup_rxnorm(drug_name: str) -> Optional[str]:
    """Return RxNorm CUI for a drug name via the public RxNorm API."""
    if drug_name in _rxnorm_cache:
        return _rxnorm_cache[drug_name]
    try:
        resp = requests.get(
            f"{RXNORM_BASE}/rxcui.json",
            params={"name": drug_name, "search": "1"},
            timeout=5,
        )
        data = resp.json()
        cui = data.get("idGroup", {}).get("rxnormId", [None])[0]
        _rxnorm_cache[drug_name] = cui
        return cui
    except Exception:
        _rxnorm_cache[drug_name] = None
        return None


# ── MeSH ──────────────────────────────────────────────────────────────────────

def _lookup_mesh(term: str) -> Optional[str]:
    """Return MeSH descriptor UI for a medical term via the NLM MeSH API."""
    if term in _mesh_cache:
        return _mesh_cache[term]
    try:
        resp = requests.get(
            f"{MESH_BASE}/descriptor",
            params={"label": term, "match": "exact", "limit": 1},
            timeout=5,
        )
        results = resp.json()
        if results:
            uid = results[0].get("resource", "").split("/")[-1]
            _mesh_cache[term] = uid or None
            return _mesh_cache[term]
    except Exception:
        pass
    # Try fuzzy fallback
    try:
        resp = requests.get(
            f"{MESH_BASE}/descriptor",
            params={"label": term, "match": "contains", "limit": 1},
            timeout=5,
        )
        results = resp.json()
        if results:
            uid = results[0].get("resource", "").split("/")[-1]
            _mesh_cache[term] = uid or None
            return _mesh_cache[term]
    except Exception:
        pass
    _mesh_cache[term] = None
    return None


# ── UMLS ──────────────────────────────────────────────────────────────────────

def _get_umls_ticket(tgt: str) -> Optional[str]:
    """Get a single-use UMLS service ticket from a TGT."""
    try:
        resp = requests.post(
            tgt,
            data={"service": "http://umlsks.nlm.nih.gov"},
            timeout=5,
        )
        return resp.text.strip()
    except Exception:
        return None


def _get_umls_tgt() -> Optional[str]:
    """Authenticate with UMLS and get a Ticket-Granting Ticket."""
    if not UMLS_API_KEY:
        return None
    try:
        resp = requests.post(
            f"{UMLS_BASE}/auth/token",
            data={"apikey": UMLS_API_KEY},
            timeout=10,
        )
        location = resp.headers.get("location", "")
        return location or None
    except Exception:
        return None


_umls_tgt: Optional[str] = None  # cached TGT for session


def _lookup_umls(term: str) -> Optional[str]:
    """Return UMLS CUI for a term (requires UMLS_API_KEY in .env)."""
    global _umls_tgt
    if not UMLS_API_KEY:
        return None
    if term in _umls_cache:
        return _umls_cache[term]
    try:
        if _umls_tgt is None:
            _umls_tgt = _get_umls_tgt()
        if _umls_tgt is None:
            return None
        ticket = _get_umls_ticket(_umls_tgt)
        if not ticket:
            return None
        resp = requests.get(
            f"{UMLS_BASE}/search/current",
            params={"string": term, "searchType": "normalizedString",
                    "ticket": ticket, "returnIdType": "concept"},
            timeout=5,
        )
        results = resp.json().get("result", {}).get("results", [])
        cui = results[0]["ui"] if results and results[0].get("ui") != "NONE" else None
        _umls_cache[term] = cui
        return cui
    except Exception:
        _umls_cache[term] = None
        return None


# ── Main linker ───────────────────────────────────────────────────────────────

# Entity types that benefit from each identifier source
_RXNORM_TYPES = {"Drug", "Compound"}
_MESH_TYPES   = {"Disease", "Symptom", "Gene", "Protein", "Organ",
                 "Pathway", "BiologicalProcess", "MolecularFunction"}
_UMLS_TYPES   = set()  # all types, used as fallback if UMLS key is set


def link_identifiers(
    entities:       List[Dict],
    rate_limit_s:   float = 0.15,
    verbose:        bool  = True,
) -> List[Dict]:
    """
    Enrich a list of entity dicts with canonical medical identifiers.

    Modifies entities in-place and also returns them.

    Args:
        entities:     Output of deduplicator.get_entities()
        rate_limit_s: Seconds between API calls (be polite to free APIs)
        verbose:      Print progress

    Returns:
        Same list with umls_cui, mesh_id, rxnorm_id added where found
    """
    found_rxnorm = found_mesh = found_umls = 0

    for i, entity in enumerate(entities):
        name  = entity["name"]
        etype = entity.get("type", "")
        ids: Dict[str, Optional[str]] = {}

        # Tier 1: RxNorm for drugs
        if etype in _RXNORM_TYPES:
            rxnorm_id = _lookup_rxnorm(name)
            if not rxnorm_id:
                # Try aliases
                for alias in entity.get("aliases", []):
                    rxnorm_id = _lookup_rxnorm(alias)
                    if rxnorm_id:
                        break
            if rxnorm_id:
                ids["rxnorm_id"] = rxnorm_id
                found_rxnorm += 1
            time.sleep(rate_limit_s)

        # Tier 2: MeSH for biomedical concepts
        if etype in _MESH_TYPES:
            mesh_id = _lookup_mesh(name)
            if not mesh_id:
                for alias in entity.get("aliases", []):
                    mesh_id = _lookup_mesh(alias)
                    if mesh_id:
                        break
            if mesh_id:
                ids["mesh_id"] = mesh_id
                found_mesh += 1
            time.sleep(rate_limit_s)

        # Tier 3: UMLS fallback (if API key available and no ID found yet)
        if UMLS_API_KEY and not ids:
            umls_cui = _lookup_umls(name)
            if umls_cui:
                ids["umls_cui"] = umls_cui
                found_umls += 1
            time.sleep(rate_limit_s)

        entity.update(ids)

        if verbose and (i + 1) % 10 == 0:
            print(f"[identifier_linker] {i+1}/{len(entities)} entities processed")

    if verbose:
        print(f"[identifier_linker] ✅ Done: "
              f"RxNorm={found_rxnorm}, MeSH={found_mesh}, UMLS={found_umls} "
              f"out of {len(entities)} entities")
    return entities


if __name__ == "__main__":
    test_entities = [
        {"type": "Drug",    "name": "Metformin",    "aliases": ["Glucophage"]},
        {"type": "Drug",    "name": "Aspirin",      "aliases": []},
        {"type": "Disease", "name": "Type 2 Diabetes", "aliases": ["T2DM"]},
        {"type": "Gene",    "name": "BRCA1",        "aliases": []},
    ]
    result = link_identifiers(test_entities, verbose=True)
    import json
    print(json.dumps(result, indent=2))
