"""
pipeline/entity_extractor_lite.py
Phase 5 — Lightweight entity spotter for query-time use.

Used in the Qdrant-first retrieval pipeline to extract entity names
from retrieved chunks before graph traversal — so graph search starts
from semantically grounded entities rather than LLM guesses from the
raw question alone.

Unlike ingestion/entity_extractor.py (which does full NER + relations),
this module only extracts entity *names and types* — fast and cheap.
"""

from __future__ import annotations
import json
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a fast biomedical entity spotter.

Given one or more text passages, identify all medical/biomedical entity names.

Entity types to look for:
  Disease, Drug, Gene, Protein, Pathway, Symptom, Treatment,
  Compound, Organ, CellType, MolecularFunction, BiologicalProcess

RULES:
- Include only explicitly mentioned entities. Do not infer.
- Normalize names to proper case (e.g. "type 2 diabetes" → "Type 2 Diabetes").
- De-duplicate: if the same entity appears multiple times, list it once.
- Return ONLY valid JSON.

OUTPUT FORMAT:
{
  "entities": [
    {"name": "Metformin", "type": "Drug"},
    {"name": "Type 2 Diabetes", "type": "Disease"},
    {"name": "AMPK", "type": "Protein"}
  ]
}"""


def _call_groq(text: str) -> dict:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Text:\n\n{text[:4000]}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=512,
    )
    return json.loads(resp.choices[0].message.content)


def _call_openai(text: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # cheaper model — entity spotting doesn't need 4o
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Text:\n\n{text[:4000]}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=512,
    )
    return json.loads(resp.choices[0].message.content)


def _scispacy_fallback(text: str) -> List[Dict]:
    """
    Zero-API fallback using scispaCy (en_core_sci_sm).
    Returns [] if scispaCy is not installed.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_sci_sm")
        doc = nlp(text[:5000])
        seen = set()
        entities = []
        for ent in doc.ents:
            name = ent.text.strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                entities.append({"name": name, "type": "Unknown"})
        return entities
    except Exception:
        return []


def spot_entities_in_chunks(
    chunks:   List[Dict],
    provider: str = "auto",
) -> List[str]:
    """
    Extract unique entity names from a list of retrieved chunk dicts.

    Args:
        chunks:   List of {text, source, page, ...} dicts from vector retrieval
        provider: "auto" | "groq" | "openai" | "scispacy"

    Returns:
        Deduplicated list of entity name strings
    """
    if not chunks:
        return []

    # Combine chunk texts (up to ~3000 chars to stay within token limits)
    combined = "\n\n---\n\n".join(
        c.get("text", "")[:600] for c in chunks[:6]
    )

    if provider == "scispacy":
        raw_entities = _scispacy_fallback(combined)
        return list({e["name"] for e in raw_entities if e["name"]})

    try:
        if provider == "groq" or (provider == "auto" and os.getenv("GROQ_API_KEY")):
            data = _call_groq(combined)
        else:
            data = _call_openai(combined)

        entities = data.get("entities", [])
        names = list({e["name"] for e in entities if e.get("name")})
        print(f"[entity_extractor_lite] Found {len(names)} entities in retrieved chunks")
        return names

    except Exception as e:
        print(f"[entity_extractor_lite] LLM failed ({e}), trying scispaCy fallback")
        raw = _scispacy_fallback(combined)
        return list({e["name"] for e in raw if e["name"]})
