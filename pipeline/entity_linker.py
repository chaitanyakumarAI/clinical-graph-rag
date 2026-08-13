"""
pipeline/entity_linker.py
Link query-time entity names to Neo4j node IDs using a 3-tier cascade.

Tier 1 — Exact / alias match    (score 1.0)  — fast dict lookup
Tier 2 — Fuzzy match            (score ≥ 0.82) — rapidfuzz partial_ratio
Tier 3 — Embedding similarity   (score ≥ 0.78) — sentence-transformers cosine

Tier 3 activates only when tiers 1+2 both fail, so the common case (exact/fuzzy)
remains fast. The embedding index is built once from all graph node names and
cached for the process lifetime.

Handles medical paraphrases that fuzzy matching misses:
  "blood thinner"   → Warfarin
  "high blood sugar"→ Hyperglycemia
  "sugar disease"   → Type 2 Diabetes Mellitus
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz, process as rfprocess

from graph.neo4j_client import Neo4jClient

FUZZY_THRESHOLD    = 82    # rapidfuzz partial_ratio threshold (0-100)
EMBED_THRESHOLD    = 0.78  # cosine similarity threshold for embedding tier
EMBED_MODEL        = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


# ── Embedding linker (tier 3) ─────────────────────────────────────────────────

class _EmbeddingLinker:
    """
    Cosine-similarity entity linker backed by sentence-transformers.
    Node name embeddings are computed once per session and cached.
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        self._model_name = model_name
        self._model      = None
        self._names:  List[str]        = []
        self._matrix: Optional[np.ndarray] = None   # shape (N, D), L2-normalised

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                print(f"[entity_linker] Embedding model '{self._model_name}' loaded")
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not found: pip install sentence-transformers"
                ) from e

    def build_index(self, names: List[str]):
        """Encode all graph node names and build the cosine-similarity index."""
        if not names:
            return
        self._load_model()
        vecs = self._model.encode(names, show_progress_bar=False, batch_size=512,
                                   normalize_embeddings=True)
        self._names  = names
        self._matrix = np.array(vecs, dtype=np.float32)

    def query(self, term: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return top_k (name, cosine_similarity) pairs for a query term."""
        if self._matrix is None or not self._names:
            return []
        self._load_model()
        vec = self._model.encode([term], show_progress_bar=False,
                                  normalize_embeddings=True)[0].astype(np.float32)
        scores = self._matrix @ vec   # cosine similarity (matrix already normalised)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._names[i], float(scores[i])) for i in top_idx]

    @property
    def ready(self) -> bool:
        return self._matrix is not None


# ── Module-level caches ───────────────────────────────────────────────────────

_node_cache:   Optional[Dict]    = None   # {lower_name → {name, label, aliases}}
_embed_linker: _EmbeddingLinker  = _EmbeddingLinker()
_embed_built:  bool              = False


def _get_all_node_names(client: Neo4jClient) -> Dict[str, Dict]:
    """Build {lower_name → node_info} from all named Neo4j nodes."""
    cypher = """
    MATCH (n)
    WHERE n.name IS NOT NULL
      AND NOT n:Chunk AND NOT n:Document AND NOT n:Evidence
    RETURN n.name    AS name,
           labels(n)[0] AS label,
           n.aliases AS aliases
    """
    rows  = client.run_query(cypher)
    index = {}
    for row in rows:
        name    = row["name"]
        aliases = row.get("aliases") or []
        entry   = {"name": name, "label": row["label"], "aliases": aliases}
        index[name.lower()] = entry
        for alias in aliases:
            if alias:
                index[alias.lower()] = entry
    return index


def _ensure_embed_index(node_cache: Dict):
    """Build the embedding index the first time it is needed."""
    global _embed_built
    if _embed_built:
        return
    # Use canonical names (de-duplicated from index values)
    canonical = list({v["name"]: v for v in node_cache.values()}.keys())
    try:
        _embed_linker.build_index(canonical)
        _embed_built = True
        print(f"[entity_linker] Embedding index built: {len(canonical)} nodes")
    except ImportError as e:
        print(f"[entity_linker] Embedding tier disabled ({e})")


# ── Public API ────────────────────────────────────────────────────────────────

def link_entities(
    entity_names:  List[str],
    client:        Neo4jClient,
    refresh_cache: bool = False,
) -> List[Dict]:
    """
    Resolve query entity names to Neo4j nodes via 3-tier cascade.

    Returns:
        List of resolved dicts:
        {query_term, neo4j_name, label, score, method}
        Unresolved entities are omitted with a warning.
    """
    global _node_cache, _embed_built

    if _node_cache is None or refresh_cache:
        _node_cache  = _get_all_node_names(client)
        _embed_built = False   # force rebuild when cache refreshes

    linked: List[Dict] = []

    for term in entity_names:
        lower = term.lower().strip()
        if not lower:
            continue

        # ── Tier 1: Exact / alias match ──────────────────────────────────────
        if lower in _node_cache:
            e = _node_cache[lower]
            linked.append({
                "query_term": term, "neo4j_name": e["name"],
                "label": e["label"], "score": 1.0, "method": "exact",
            })
            continue

        # ── Tier 2: Fuzzy match ───────────────────────────────────────────────
        candidates = list(_node_cache.keys())
        result = rfprocess.extractOne(lower, candidates, scorer=fuzz.partial_ratio)
        if result and result[1] >= FUZZY_THRESHOLD:
            e = _node_cache[result[0]]
            linked.append({
                "query_term": term, "neo4j_name": e["name"],
                "label": e["label"], "score": result[1] / 100.0, "method": "fuzzy",
            })
            continue

        # ── Tier 3: Embedding similarity ─────────────────────────────────────
        try:
            _ensure_embed_index(_node_cache)
            if _embed_linker.ready:
                top = _embed_linker.query(term, top_k=3)
                for cand_name, sim in top:
                    if sim >= EMBED_THRESHOLD:
                        # Map back to canonical entry via the name cache
                        e = _node_cache.get(cand_name.lower())
                        if e:
                            linked.append({
                                "query_term": term, "neo4j_name": e["name"],
                                "label": e["label"],
                                "score": round(sim, 4), "method": "embedding",
                            })
                            break
                else:
                    # No tier matched
                    print(f"[entity_linker] ⚠️  Unresolved after 3 tiers: '{term}'")
                    continue
            else:
                print(f"[entity_linker] ⚠️  Embedding not ready, skipping: '{term}'")
        except Exception as e_err:
            print(f"[entity_linker] Embedding tier error for '{term}': {e_err}")

    n_resolved = len(linked)
    n_total    = len([t for t in entity_names if t.strip()])
    methods    = {}
    for r in linked:
        methods[r["method"]] = methods.get(r["method"], 0) + 1
    method_str = ", ".join(f"{m}={c}" for m, c in methods.items())
    print(f"[entity_linker] {n_resolved}/{n_total} linked "
          f"({method_str or 'none'})")
    return linked


def invalidate_cache():
    """Force rebuild of node name cache and embedding index on next call."""
    global _node_cache, _embed_built
    _node_cache  = None
    _embed_built = False
