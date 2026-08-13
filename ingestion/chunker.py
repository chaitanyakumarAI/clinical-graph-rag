"""
ingestion/chunker.py
Stage 1 — Semantic sliding-window chunker.

Splits page text into overlapping chunks of ~512 tokens.
Each chunk carries full provenance (source, page, chunk_id).
"""

from __future__ import annotations
import hashlib
from typing import List, Dict

# Approximate token count: 1 token ≈ 4 chars (conservative for medical text)
CHARS_PER_TOKEN = 4
CHUNK_TOKENS    = 512
OVERLAP_TOKENS  = 100


def _token_count(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _make_chunk_id(source: str, page: int, idx: int) -> str:
    raw = f"{source}_p{page}_c{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_pages(pages: List[Dict], chunk_tokens: int = CHUNK_TOKENS,
                overlap_tokens: int = OVERLAP_TOKENS) -> List[Dict]:
    """
    Split parsed pages into overlapping text chunks.

    Args:
        pages: Output of pdf_parser.parse_pdf()
        chunk_tokens: Target chunk size in tokens
        overlap_tokens: Overlap between consecutive chunks

    Returns:
        List of chunk dicts:
        {
          "chunk_id": str,
          "text": str,
          "source": str,
          "page": int,
          "chunk_index": int,
          "token_count": int,
        }
    """
    chunk_chars   = chunk_tokens   * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    step_chars    = chunk_chars - overlap_chars

    chunks: List[Dict] = []

    for page_data in pages:
        text   = page_data["text"]
        source = page_data["source"]
        page   = page_data["page"]

        if len(text) <= chunk_chars:
            # Entire page fits in one chunk
            chunk_id = _make_chunk_id(source, page, 0)
            chunks.append({
                "chunk_id":    chunk_id,
                "text":        text,
                "source":      source,
                "page":        page,
                "chunk_index": 0,
                "token_count": _token_count(text),
            })
        else:
            idx   = 0
            start = 0
            while start < len(text):
                end      = min(start + chunk_chars, len(text))
                chunk_text = text[start:end].strip()

                if len(chunk_text) > 100:  # skip tiny trailing fragments
                    chunk_id = _make_chunk_id(source, page, idx)
                    chunks.append({
                        "chunk_id":    chunk_id,
                        "text":        chunk_text,
                        "source":      source,
                        "page":        page,
                        "chunk_index": idx,
                        "token_count": _token_count(chunk_text),
                    })
                    idx += 1

                if end == len(text):
                    break
                start += step_chars

    print(f"[chunker] {len(pages)} pages → {len(chunks)} chunks "
          f"(~{chunk_tokens} tok, {overlap_tokens} tok overlap)")
    return chunks


if __name__ == "__main__":
    import sys, json
    from ingestion.pdf_parser import parse_pdf
    pages  = parse_pdf(sys.argv[1])
    chunks = chunk_pages(pages)
    print(json.dumps(chunks[:3], indent=2))
    print(f"Total chunks: {len(chunks)}")
