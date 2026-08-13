"""
ingestion/entity_extractor.py
Stage 1 — LLM-assisted entity and relation extraction from text chunks.

Defaults to Groq (Llama 3.3-70B, free tier).
Falls back to OpenAI GPT-4o if GROQ_API_KEY is absent.

Phase 1 enhancements:
  - Expanded entity schema: +Gene, Protein, Pathway, Compound, CellType,
    MolecularFunction, BiologicalProcess
  - Entity-level confidence scores (not just relation-level)
  - Expanded relation schema: +TARGETS, REGULATES, INHIBITS, ACTIVATES,
    LEADS_TO, BIOMARKER_FOR, EXPRESSED_IN, PARTICIPATES_IN
  - ExtractionResult now carries page and paragraph_index for full provenance
"""

from __future__ import annotations
import json
import os
import time
from typing import List, Dict, Optional
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

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

from ingestion.ontology_validator import validate_triple as _validate_triple
import re as _re


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
    words = [w for w in _re.split(r'\W+', name.lower()) if len(w) >= 4]
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


# ── LLM Clients ───────────────────────────────────────────────────────────────

def _call_groq(text: str) -> dict:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=4096,
    )
    return json.loads(response.choices[0].message.content)


def _call_openai(text: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=4096,
    )
    return json.loads(response.choices[0].message.content)


def _call_llm(text: str, provider: str = "auto") -> dict:
    """Call LLM with auto-detection of available provider."""
    if provider == "groq" or (provider == "auto" and os.getenv("GROQ_API_KEY")):
        return _call_groq(text)
    elif provider == "openai" or (provider == "auto" and os.getenv("OPENAI_API_KEY")):
        return _call_openai(text)
    else:
        raise EnvironmentError(
            "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in .env"
        )


# ── Main Extractor ────────────────────────────────────────────────────────────

def extract_from_chunk(
    chunk: Dict,
    provider: str = "auto",
    retry: int = 2,
    delay: float = 1.0,
) -> ExtractionResult:
    """
    Run LLM extraction on a single chunk dict.

    Args:
        chunk:    A chunk dict from chunker.chunk_pages()
        provider: "auto" | "groq" | "openai"
        retry:    Number of retries on failure
        delay:    Seconds between retries

    Returns:
        ExtractionResult with validated entities and relations, plus provenance
    """
    text            = chunk["text"]
    chunk_id        = chunk["chunk_id"]
    source          = chunk["source"]
    page            = chunk.get("page", 0)
    chunk_index     = chunk.get("chunk_index", 0)

    for attempt in range(retry + 1):
        try:
            raw = _call_llm(text, provider)
            entities_raw  = []
            relations_raw = []

            for e in raw.get("entities", []):
                try:
                    entities_raw.append(ExtractedEntity(**e))
                except Exception:
                    pass

            for r in raw.get("relations", []):
                try:
                    relations_raw.append(ExtractedRelation(**r))
                except Exception:
                    pass

            # ── Validation pass ───────────────────────────────────────────────
            entities: List[ExtractedEntity] = []
            for ent in entities_raw:
                # 1. Confidence threshold
                if ent.confidence < MIN_ENTITY_CONFIDENCE:
                    continue
                # 2. Text grounding — name (or alias) must appear in chunk
                if not _is_grounded(ent.name, ent.aliases, text):
                    continue
                entities.append(ent)

            valid_names = {e.name.lower() for e in entities}
            relations: List[ExtractedRelation] = []
            for rel in relations_raw:
                # 3. Confidence threshold
                if rel.confidence < MIN_RELATION_CONFIDENCE:
                    continue
                # 4. Type-predicate consistency
                if not _is_type_consistent(rel):
                    continue
                relations.append(rel)

            # Stats
            n_ent_raw  = len(entities_raw)
            n_rel_raw  = len(relations_raw)
            n_ent_kept = len(entities)
            n_rel_kept = len(relations)
            if n_ent_raw > n_ent_kept or n_rel_raw > n_rel_kept:
                print(f"  [validate] entities {n_ent_raw}→{n_ent_kept}, "
                      f"relations {n_rel_raw}→{n_rel_kept}")

            return ExtractionResult(
                entities        = entities,
                relations       = relations,
                chunk_id        = chunk_id,
                source          = source,
                page            = page,
                paragraph_index = chunk_index,
            )

        except Exception as exc:
            if attempt < retry:
                time.sleep(delay * (attempt + 1))
            else:
                print(f"[extractor] Failed chunk {chunk_id}: {exc}")
                return ExtractionResult(
                    chunk_id=chunk_id, source=source,
                    page=page, paragraph_index=chunk_index,
                )


def extract_from_chunks(
    chunks: List[Dict],
    provider: str = "auto",
    rate_limit_delay: float = 0.5,
) -> List[ExtractionResult]:
    """
    Run extraction on all chunks with progress reporting.

    Args:
        chunks:           Output of chunker.chunk_pages()
        provider:         LLM provider
        rate_limit_delay: Seconds between API calls (avoid rate limits)

    Returns:
        List of ExtractionResult, one per chunk
    """
    results = []
    total   = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        print(f"[extractor] Chunk {i}/{total} (id={chunk['chunk_id']}, "
              f"page={chunk.get('page',0)}, ~{chunk.get('token_count',0)} tok)", end=" ")
        result = extract_from_chunk(chunk, provider=provider)
        print(f"→ {len(result.entities)} entities, {len(result.relations)} relations")
        results.append(result)

        if i < total:
            time.sleep(rate_limit_delay)

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
    result = extract_from_chunk(sample)
    import json as _json
    print(_json.dumps(result.model_dump(), indent=2))
