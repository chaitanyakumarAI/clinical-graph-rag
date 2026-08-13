"""
ingestion/pdf_parser.py
Stage 1 — Parse medical PDFs into per-page text using PyMuPDF.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF


def parse_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from each page of a PDF.

    Returns:
        List of dicts: [{"page": int, "text": str, "source": str, "total_pages": int}]
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    source_name = path.stem  # filename without extension
    pages = []

    doc = fitz.open(str(path))
    total = len(doc)

    for page_num, page in enumerate(doc, start=1):
        # Extract text with layout preservation
        text = page.get_text("text").strip()

        # Skip nearly-empty pages (headers/footers only)
        if len(text) < 50:
            continue

        pages.append({
            "page": page_num,
            "text": text,
            "source": source_name,
            "total_pages": total,
        })

    doc.close()
    print(f"[pdf_parser] '{source_name}': {len(pages)}/{total} pages extracted")
    return pages


def parse_pdf_directory(pdf_dir: str) -> List[Dict]:
    """Parse all PDFs in a directory and combine results."""
    dir_path = Path(pdf_dir)
    all_pages = []
    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDFs found in: {dir_path}")

    for pdf_file in pdf_files:
        all_pages.extend(parse_pdf(str(pdf_file)))

    print(f"[pdf_parser] Total pages from {len(pdf_files)} PDFs: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    import sys, json
    pages = parse_pdf(sys.argv[1])
    print(json.dumps(pages[:2], indent=2))
