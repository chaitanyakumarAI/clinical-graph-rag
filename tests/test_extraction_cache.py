"""ingestion/entity_extractor.py — validation filters and the per-chunk cache."""

import pytest

from ingestion import entity_extractor as ex

TEXT = ("Metformin is first-line treatment for Type 2 Diabetes. "
        "It can cause lactic acidosis in patients with renal impairment.")
CHUNK = {"chunk_id": "c1", "source": "guide", "page": 3, "chunk_index": 0, "text": TEXT}
REPLY = {
    "entities": [
        {"type": "Drug", "name": "Metformin", "confidence": 0.98},
        {"type": "Disease", "name": "Type 2 Diabetes", "aliases": ["T2DM"], "confidence": 0.97},
        {"type": "SideEffect", "name": "Lactic Acidosis", "confidence": 0.9},
        {"type": "Drug", "name": "Insulin Glargine", "confidence": 0.95},       # not in text
        {"type": "Planet", "name": "Mars", "confidence": 0.99},                  # unknown type
        {"type": "Drug", "name": "Renal", "confidence": 0.2},                    # low confidence
    ],
    "relations": [
        {"subject_type": "Drug", "subject": "Metformin", "predicate": "treats",
         "object_type": "Disease", "object": "Type 2 Diabetes", "confidence": 0.95},
        {"subject_type": "Drug", "subject": "Metformin", "predicate": "CAUSES",
         "object_type": "SideEffect", "object": "Lactic Acidosis", "confidence": 0.9},
        {"subject_type": "Disease", "subject": "Type 2 Diabetes", "predicate": "TREATS",    # wrong direction
         "object_type": "Drug", "object": "Metformin", "confidence": 0.9},
        {"subject_type": "Drug", "subject": "Metformin", "predicate": "CAUSES",
         "object_type": "SideEffect", "object": "Nausea", "confidence": 0.3},            # hedged
    ],
}


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("EXTRACTION_MODEL", "model-a")
    return tmp_path


def test_validation_drops_ungrounded_unknown_and_invalid_items(cache_dir, monkeypatch, fake_llm):
    monkeypatch.setattr(ex, "chat_json", fake_llm(REPLY))

    result = ex.extract_from_chunk(CHUNK)

    assert [e.name for e in result.entities] == ["Metformin", "Type 2 Diabetes", "Lactic Acidosis"]
    assert [(r.subject, r.predicate, r.object) for r in result.relations] == [
        ("Metformin", "TREATS", "Type 2 Diabetes"),          # predicate normalised to upper case
        ("Metformin", "CAUSES", "Lactic Acidosis"),
    ]
    assert (result.chunk_id, result.page) == ("c1", 3)


def test_second_run_is_served_from_cache(cache_dir, monkeypatch, fake_llm):
    llm = fake_llm(REPLY)                     # only one reply available
    monkeypatch.setattr(ex, "chat_json", llm)

    first = ex.extract_from_chunk(CHUNK)
    second = ex.extract_from_chunk(CHUNK)     # would raise if it called the LLM again

    assert len(llm.calls) == 1
    assert second.entities == first.entities and second.relations == first.relations
    assert len(list(cache_dir.iterdir())) == 1


def test_changing_the_model_invalidates_the_cache(cache_dir, monkeypatch, fake_llm):
    llm = fake_llm(REPLY, {"entities": [], "relations": []})
    monkeypatch.setattr(ex, "chat_json", llm)

    ex.extract_from_chunk(CHUNK)
    monkeypatch.setenv("EXTRACTION_MODEL", "model-b")
    result = ex.extract_from_chunk(CHUNK)

    assert len(llm.calls) == 2                # stale model-a result not reused
    assert result.entities == []


def test_fingerprint_tracks_prompt_and_thresholds(cache_dir, monkeypatch):
    base = ex._cache_fingerprint()
    assert ex._cache_fingerprint() == base                       # deterministic

    monkeypatch.setattr(ex, "SYSTEM_PROMPT", ex.SYSTEM_PROMPT + "\nNew rule.")
    assert ex._cache_fingerprint() != base

    monkeypatch.undo()
    monkeypatch.setattr(ex, "CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("EXTRACTION_MODEL", "model-a")
    assert ex._cache_fingerprint() == base
    monkeypatch.setattr(ex, "MIN_RELATION_CONFIDENCE", 0.7)
    assert ex._cache_fingerprint() != base


def test_cache_path_depends_on_text_and_fingerprint():
    assert ex._cache_path("abc", "f1") == ex._cache_path("  abc \n", "f1")   # whitespace-insensitive
    assert ex._cache_path("abc", "f1") != ex._cache_path("abd", "f1")
    assert ex._cache_path("abc", "f1") != ex._cache_path("abc", "f2")


def test_malformed_reply_is_retried_then_skipped_without_caching(cache_dir, monkeypatch, fake_llm):
    llm = fake_llm(ValueError("bad json"), ValueError("bad json again"))
    monkeypatch.setattr(ex, "chat_json", llm)

    result = ex.extract_from_chunk(CHUNK)

    assert len(llm.calls) == 2 and result.entities == []
    assert list(cache_dir.iterdir()) == []     # a failure must not be cached as "nothing found"
