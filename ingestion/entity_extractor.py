"""
ingestion/entity_extractor.py
Stage 1 — LLM-assisted entity and relation extraction from text chunks.

The LLM provider (Gemini / Groq / OpenAI / local Ollama) is chosen in
llm/client.py, which also handles rate limiting and retries.

Long PDFs on free-tier APIs:
  - Every chunk's result is cached on disk (.cache/extractions/), so if a run
    stops because the daily quota is used up, re-running resumes where it
    stopped instead of paying for the same chunks again.
  - Set EXTRACTION_MODEL to use a cheaper/higher-quota model just for bulk
    extraction (e.g. llama-3.1-8b-instant on Groq) while answers still use
    the default model.

Phase 1 enhancements:
  - Expanded entity schema: +Gene, Protein, Pathway, Compound, CellType,
    MolecularFunction, BiologicalProcess
  - Entity-level confidence scores (not just relation-level)
  - Expanded relation schema: +TARGETS, REGULATES, INHIBITS, ACTIVATES,
    LEADS_TO, BIOMARKER_FOR, EXPRESSED_IN, PARTICIPATES_IN
  - ExtractionResult now carries page and paragraph_index for full provenance
"""

from __future__ import annotations
import hashlib
import json
import os
import re
from typing import List, Dict, Optional

import openai
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from ingestion.ontology_validator import validate_triple as _validate_triple
from llm import RateLimitExhausted, chat_json

load_dotenv()

# ── Schema ────────────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = {
    # Clinical
    "Disease", "Drug", "Symptom", "Treatment",
    "SideEffect", "PatientGroup", "Organ", "MedicalGuideline",
    # Molecular / Biomedical
    "Gene", "Protein", "Pathway", "Compound",
    "CellType", "MolecularFunction", "BiologicalProcess",
}

VALID_RELATION_TYPES = {
    # Clinical relations
    "TREATS", "CAUSES", "INTERACTS_WITH", "HAS_SYMPTOM",
    "CONTRAINDICATED_FOR", "AFFECTS_ORGAN", "RECOMMENDED_BY",
    # Molecular relations
    "TARGETS", "REGULATES", "INHIBITS", "ACTIVATES",
    "LEADS_TO", "BIOMARKER_FOR", "EXPRESSED_IN", "PARTICIPATES_IN",
    # Fallback
    "RELATED_TO",
}


class ExtractedEntity(BaseModel):
    type: str
    name: str
    aliases: List[str] = []
    confidence: float = 0.8   # Phase 1: entity-level confidence

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        if v not in VALID_ENTITY_TYPES:
            raise ValueError(f"Unknown entity type: {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))


class ExtractedRelation(BaseModel):
    subject_type: str
    subject: str
    predicate: str
    object_type: str
    object: str
    confidence: float = 0.8

    @field_validator("predicate")
    @classmethod
    def validate_predicate(cls, v):
        v = v.upper().replace(" ", "_")
        if v not in VALID_RELATION_TYPES:
            v = "RELATED_TO"  # graceful fallback
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))


class ExtractionResult(BaseModel):
    entities:        List[ExtractedEntity]  = []
    relations:       List[ExtractedRelation] = []
    chunk_id:        str = ""
    source:          str = ""
    page:            int = 0          # Phase 1: provenance
    paragraph_index: int = 0          # Phase 1: provenance


# ── Validation ────────────────────────────────────────────────────────────────

# Minimum confidence thresholds — entities/relations below these are dropped
MIN_ENTITY_CONFIDENCE   = 0.40
MIN_RELATION_CONFIDENCE = 0.50


def _is_grounded(name: str, aliases: List[str], text: str) -> bool:
    """
    Return True if the entity name (or any alias) appears in the source text.
    Prevents hallucinated entities from entering the knowledge graph.

    Checks (in order):
      1. Exact substring match (case-insensitive)
      2. Alias substring match
      3. All significant words of the name appear (for long multi-word names)
    """
    text_l = text.lower()
    if name.lower() in text_l:
        return True
    for alias in aliases:
        if alias and alias.lower() in text_l:
            return True
    # Fuzzy: require all significant words (≥4 chars) to appear
    words = [w for w in re.split(r'\W+', name.lower()) if len(w) >= 4]
    if len(words) >= 2 and all(w in text_l for w in words):
        return True
    return False


def _is_type_consistent(rel: "ExtractedRelation") -> bool:
    """
    Delegate to the central ontology validator.
    Returns True if the (subject_type, predicate, object_type) triple is valid.
    Missing types always pass (conservative — don't block when LLM omits types).
    """
    result = _validate_triple(rel.subject_type, rel.predicate, rel.object_type)
    if not result.is_valid and result.suggested_predicate:
        # Direction error — log but still drop (extractor shouldn't silently swap args)
        print(f"  [ontology] direction reversed: {rel.subject_type}→{rel.predicate}→{rel.object_type}")
    return result.is_valid



SYSTEM_PROMPT = """You are a biomedical NLP expert. Extract medical entities and relationships from the text.

ENTITY TYPES (use exactly these labels):
Clinical:   Disease, Drug, Symptom, Treatment, SideEffect, PatientGroup, Organ, MedicalGuideline
Molecular:  Gene, Protein, Pathway, Compound, CellType, MolecularFunction, BiologicalProcess

RELATION TYPES (use exactly these labels):
Clinical:
  TREATS              - Drug/Treatment treats Disease
  CAUSES              - Drug/Disease causes SideEffect/Disease
  INTERACTS_WITH      - Drug interacts with Drug
  HAS_SYMPTOM         - Disease has Symptom
  CONTRAINDICATED_FOR - Drug contraindicated for PatientGroup/Disease
  AFFECTS_ORGAN       - Drug/Disease affects Organ
  RECOMMENDED_BY      - Treatment recommended by MedicalGuideline

Molecular:
  TARGETS             - Drug/Compound targets Protein/Gene
  REGULATES           - Gene/Protein regulates Gene/Protein/Pathway
  INHIBITS            - Drug/Compound/Protein inhibits Gene/Protein
  ACTIVATES           - Drug/Compound/Protein activates Gene/Protein/Pathway
  LEADS_TO            - Disease/Pathway leads to Disease/Pathway
  BIOMARKER_FOR       - Gene/Protein is biomarker for Disease
  EXPRESSED_IN        - Gene/Protein expressed in CellType/Organ
  PARTICIPATES_IN     - Gene/Protein/Compound participates in Pathway/BiologicalProcess

Fallback:
  RELATED_TO          - General semantic relation (use only when no other type fits)

CONFIDENCE SCORING:
- Entities: 0.95+ explicitly named; 0.75-0.95 implied/inferred; <0.75 uncertain
- Relations: 0.9+ explicit statement; 0.7-0.9 strongly implied; <0.7 uncertain/hedged
  Example hedged language: "may interact", "possibly associated", "suggested to"

RULES:
- Extract only factual information present in the text. Do NOT hallucinate.
- Normalize names: proper case, full names preferred (e.g. "Type 2 Diabetes" not "T2DM").
- Include common synonyms/abbreviations in the aliases list.
- Return ONLY valid JSON, no explanation text.

OUTPUT FORMAT:
{
  "entities": [
    {"type": "Drug", "name": "Metformin", "aliases": ["metformin hydrochloride", "Glucophage"], "confidence": 0.98},
    {"type": "Disease", "name": "Type 2 Diabetes", "aliases": ["T2DM", "diabetes mellitus type 2"], "confidence": 0.99},
    {"type": "Protein", "name": "AMPK", "aliases": ["AMP-activated protein kinase"], "confidence": 0.92}
  ],
  "relations": [
    {"subject_type": "Drug", "subject": "Metformin", "predicate": "TREATS",
     "object_type": "Disease", "object": "Type 2 Diabetes", "confidence": 0.99},
    {"subject_type": "Drug", "subject": "Metformin", "predicate": "ACTIVATES",
     "object_type": "Protein", "object": "AMPK", "confidence": 0.95}
  ]
}"""

USER_PROMPT_TEMPLATE = """Extract all medical entities and relationships from this text:

---

## {text}

Return ONLY valid JSON matching the schema. If nothing relevant is found, return {{"entities": [], "relations": []}}."""

# Output budget per chunk. Providers count max_tokens against the per-minute
# token limit, so a smaller value lets more chunks through per minute.
EXTRACTION_MAX_TOKENS = int(os.getenv("EXTRACTION_MAX_TOKENS", 2048))

# Retries for a malformed (non-JSON) reply. Rate limits are retried in llm/.
MALFORMED_REPLY_RETRIES = 1


# ── Disk cache (lets an interrupted long-PDF run resume for free) ─────────────

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "extractions"
)


def _cache_path(text: str) -> str:
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{digest}.json")


def _load_cached(text: str) -> Optional[dict]:
    """Return the cached {"entities": [...], "relations": [...]} for this text, if any."""
    path = _cache_path(text)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_cached(text: str, entities: List[ExtractedEntity],
                 relations: List[ExtractedRelation]) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(text), "w", encoding="utf-8") as f:
            json.dump({
                "entities":  [e.model_dump() for e in entities],
                "relations": [r.model_dump() for r in relations],
            }, f, indent=2)
    except OSError as e:
        print(f"  [cache] warning: could not save cache: {e}")


# ── LLM call + validation ─────────────────────────────────────────────────────

def _call_llm(text: str, provider: str = "auto") -> dict:
    return chat_json(
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE.format(text=text),
        provider=provider,
        model=os.getenv("EXTRACTION_MODEL") or None,
        max_tokens=EXTRACTION_MAX_TOKENS,
    )


def _parse_and_validate(raw: dict, text: str):
    """Turn raw LLM JSON into validated entities and relations."""
    entities_raw: List[ExtractedEntity] = []
    for e in raw.get("entities", []):
        try:
            entities_raw.append(ExtractedEntity(**e))
        except Exception:
            pass   # unknown type / malformed item — skip it

    relations_raw: List[ExtractedRelation] = []
    for r in raw.get("relations", []):
        try:
            relations_raw.append(ExtractedRelation(**r))
        except Exception:
            pass

    entities = [
        ent for ent in entities_raw
        if ent.confidence >= MIN_ENTITY_CONFIDENCE          # 1. confidence threshold
        and _is_grounded(ent.name, ent.aliases, text)       # 2. must appear in the text
    ]
    relations = [
        rel for rel in relations_raw
        if rel.confidence >= MIN_RELATION_CONFIDENCE        # 3. confidence threshold
        and _is_type_consistent(rel)                        # 4. ontology-valid triple
    ]

    if len(entities) < len(entities_raw) or len(relations) < len(relations_raw):
        print(f"  [validate] entities {len(entities_raw)}→{len(entities)}, "
              f"relations {len(relations_raw)}→{len(relations)}")
    return entities, relations


# ── Main Extractor ────────────────────────────────────────────────────────────

def extract_from_chunk(
    chunk:     Dict,
    provider:  str = "auto",
    use_cache: bool = True,
) -> ExtractionResult:
    """
    Run LLM extraction on a single chunk dict.

    Args:
        chunk:     A chunk dict from chunker.chunk_pages()
        provider:  "auto" | "gemini" | "groq" | "openai" | "ollama"
        use_cache: Reuse / store results in .cache/extractions/

    Returns:
        ExtractionResult with validated entities and relations, plus provenance.
        Empty if the LLM keeps returning unusable JSON for this chunk.

    Raises:
        RateLimitExhausted: the provider's quota is used up — stop the whole run.
        openai.APIError:    e.g. invalid API key or unknown model — also fatal.
    """
    text   = chunk["text"]
    result = ExtractionResult(
        chunk_id        = chunk["chunk_id"],
        source          = chunk["source"],
        page            = chunk.get("page", 0),
        paragraph_index = chunk.get("chunk_index", 0),
    )

    cached = _load_cached(text) if use_cache else None
    if cached is not None:
        result.entities  = [ExtractedEntity(**e)   for e in cached.get("entities", [])]
        result.relations = [ExtractedRelation(**r) for r in cached.get("relations", [])]
        return result

    # Only a bad reply (invalid / truncated JSON) is worth retrying here.
    # Anything else — exhausted quota, wrong API key, unknown model — would
    # fail the same way for every chunk, so it propagates and stops the run.
    for attempt in range(MALFORMED_REPLY_RETRIES + 1):
        try:
            raw = _call_llm(text, provider)
            break
        except (ValueError, openai.BadRequestError) as exc:
            if attempt == MALFORMED_REPLY_RETRIES:
                print(f"\n  [extractor] Skipping chunk {result.chunk_id}: {exc}")
                return result
            print(f"\n  [extractor] Bad reply for chunk {result.chunk_id} ({exc}), retrying …")

    result.entities, result.relations = _parse_and_validate(raw, text)
    if use_cache:
        _save_cached(text, result.entities, result.relations)
    return result


def extract_from_chunks(
    chunks:    List[Dict],
    provider:  str = "auto",
    use_cache: bool = True,
) -> List[ExtractionResult]:
    """
    Run extraction on all chunks with progress reporting.

    Pacing is automatic (see llm/rate_limiter.py), so there is no fixed delay
    between calls and cached chunks cost nothing.

    Raises:
        RateLimitExhausted: quota used up. Chunks finished so far are cached,
                            so simply re-running later resumes from here.
    """
    results: List[ExtractionResult] = []
    total = len(chunks)
    print(f"[extractor] Extracting {total} chunks (provider={provider}, cache={use_cache})")

    for i, chunk in enumerate(chunks, 1):
        print(f"[extractor] Chunk {i}/{total} (id={chunk['chunk_id']}, "
              f"page={chunk.get('page', 0)}, ~{chunk.get('token_count', 0)} tok)", end=" ")
        try:
            result = extract_from_chunk(chunk, provider=provider, use_cache=use_cache)
        except RateLimitExhausted:
            print(f"\n[extractor] ⛔ Stopped at chunk {i}/{total}: rate limit exhausted. "
                  f"{i - 1} chunks are cached — run again later to resume.")
            raise
        print(f"→ {len(result.entities)} entities, {len(result.relations)} relations")
        results.append(result)

    total_entities  = sum(len(r.entities)  for r in results)
    total_relations = sum(len(r.relations) for r in results)
    print(f"\n[extractor] Done: {total_entities} entities, {total_relations} relations "
          f"from {total} chunks")
    return results


if __name__ == "__main__":
    # Quick test on a hardcoded sample covering both clinical and molecular entities
    sample = {
        "chunk_id":    "test_001",
        "page":        1,
        "chunk_index": 0,
        "token_count": 80,
        "source":      "test",
        "text": (
            "Metformin is the first-line treatment for Type 2 Diabetes. "
            "It activates AMPK (AMP-activated protein kinase), which inhibits "
            "hepatic glucose production. Metformin can cause GI upset and, rarely, "
            "lactic acidosis in patients with renal impairment. It is contraindicated "
            "for patients with severe chronic kidney disease."
        ),
    }
    result = extract_from_chunk(sample, use_cache=False)
    print(json.dumps(result.model_dump(), indent=2))
