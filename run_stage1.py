"""
run_stage1.py — Stage 1 CLI entry point.

Usage:
    python run_stage1.py --pdf data/pdfs/my_guideline.pdf
    python run_stage1.py --pdf-dir data/pdfs/
    python run_stage1.py --pdf data/pdfs/doc.pdf --provider groq --no-qdrant
    python run_stage1.py --pdf data/pdfs/long.pdf --chunk-tokens 1024   # fewer LLM calls

Pipeline:
    PDF → parse → chunk → LLM extract → deduplicate → [embed → Qdrant] → Neo4j

If the LLM rate limit is used up mid-run, the script stops before writing
anything. Finished chunks are cached, so re-running the same command resumes.
"""

from __future__ import annotations
import argparse
import json
import sys
import os
import time
from pathlib import Path

# Make imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

import openai

from ingestion.pdf_parser    import parse_pdf
from ingestion.chunker       import chunk_pages
from ingestion.entity_extractor import extract_from_chunks
from ingestion.deduplicator  import deduplicate
from graph.neo4j_client      import Neo4jClient
from graph.graph_loader      import load_to_graph, load_chunks_to_qdrant
from graph.document_registry import compute_sha256, is_already_ingested, register_document
from llm                     import RateLimitExhausted, model_for, resolve_provider


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Build Neo4j knowledge graph from medical PDFs"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf",     type=str, help="Path to a single PDF file")
    src.add_argument("--pdf-dir", type=str, help="Directory of PDFs to process")

    parser.add_argument(
        "--provider", choices=["auto", "gemini", "groq", "openai", "ollama"], default="auto",
        help="LLM provider for entity extraction (default: auto-detect from .env)"
    )
    parser.add_argument(
        "--chunk-tokens", type=int, default=512,
        help="Target chunk size in tokens (default: 512; 1024 halves the LLM calls for long PDFs)"
    )
    parser.add_argument(
        "--overlap-tokens", type=int, default=100,
        help="Chunk overlap in tokens (default: 100)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass and do not use disk-backed chunk extraction cache"
    )
    parser.add_argument(
        "--no-qdrant", action="store_true",
        help="Skip Qdrant embedding storage (run Stage 1 without Qdrant)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Extract entities but do NOT write to Neo4j or Qdrant"
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Save extracted entities/relations to a JSON file for inspection"
    )
    return parser.parse_args()


def _pdf_paths(args) -> list[Path]:
    if args.pdf:
        return [Path(args.pdf)]
    paths = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not paths:
        print(f"❌ No PDFs found in: {args.pdf_dir}")
        sys.exit(1)
    return paths


def _already_ingested(docs: list[dict]) -> set[str]:
    """sha256 of the docs already in Neo4j (same check as POST /ingest)."""
    try:
        with Neo4jClient() as neo4j:
            return {d["sha256"] for d in docs if is_already_ingested(d["sha256"], neo4j)}
    except Exception as e:
        print(f"\n❌ Neo4j error: {e}")
        print("   Is Neo4j running? Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env")
        sys.exit(1)


def main():
    args = parse_args()
    start = time.time()

    print("=" * 65)
    print("  Stage 1: Base Graph — PDF → Neo4j")
    print("=" * 65)

    # Each PDF is its own Document (sha256 of the file bytes + filename), exactly
    # like POST /ingest. Citations map chunk sources to Document.filename, so
    # putting several PDFs under one Document would cite the wrong file.

    # ── Step 1: Parse each PDF ──────────────────────────────────────────────
    print("\n[1/5] Parsing PDF(s) …")
    docs = []
    for path in _pdf_paths(args):
        pages = parse_pdf(str(path))
        if not pages:
            print(f"⚠️  No text extracted from {path.name}, skipping")
            continue
        docs.append({"path": path, "filename": path.name, "pages": pages,
                     "sha256": compute_sha256(path.read_bytes())})
    if not docs:
        print("❌ No pages extracted. Check your PDF path.")
        sys.exit(1)

    if not args.dry_run:
        done = _already_ingested(docs)
        for d in docs:
            if d["sha256"] in done:
                print(f"⏭️  {d['filename']} is already ingested (same sha256), skipping")
        docs = [d for d in docs if d["sha256"] not in done]
        if not docs:
            print("\nNothing to do: every PDF is already ingested.")
            return

    print("\n[2/5] Chunking …")
    for d in docs:
        d["chunks"] = chunk_pages(d["pages"], args.chunk_tokens, args.overlap_tokens)

    # ── Step 3: LLM Entity Extraction (all PDFs before any write) ─────────
    provider = resolve_provider(args.provider)
    print(f"\n[3/5] Extracting entities via LLM ({provider}: {model_for(provider)}) …")
    try:
        for d in docs:
            d["results"] = extract_from_chunks(
                d["chunks"],
                provider=provider,
                use_cache=not args.no_cache,
            )
    except RateLimitExhausted as e:
        print(f"\n⛔ {e}")
        print("   Nothing was written. Re-run the same command later to resume —")
        print("   finished chunks are cached. Or switch to a model with a bigger quota")
        print("   (see 'Rate limits' in STARTUP.md).")
        sys.exit(2)
    except openai.APIStatusError as e:
        print(f"\n❌ LLM provider error ({e.status_code}): {e.message}")
        print("   Check the API key / model name for this provider in .env.")
        sys.exit(1)

    # ── Step 4: Deduplicate (per document; shared entities merge in Neo4j) ─
    print("\n[4/5] Deduplicating …")
    for d in docs:
        d["entities"], d["relations"] = deduplicate(d["results"])

    # ── Optional: save JSON for inspection ────────────────────────────────
    if args.output_json:
        out = {"entities":  [e for d in docs for e in d["entities"]],
               "relations": [r for d in docs for r in d["relations"]]}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n📄 Saved extraction to: {args.output_json}")

    if args.dry_run:
        print("\n🔍 Dry run — skipping Qdrant and Neo4j writes.")
        print(f"   Entities extracted: {sum(len(d['entities']) for d in docs)}")
        print(f"   Relations extracted: {sum(len(d['relations']) for d in docs)}")
        elapsed = time.time() - start
        print(f"\n⏱️  Total time: {elapsed:.1f}s")
        return

    # ── Step 5: Embed + Qdrant, then Neo4j, one document at a time ─────────
    print("\n[5/5] Writing to Qdrant and Neo4j …")
    try:
        with Neo4jClient() as neo4j:
            status = neo4j.verify_connection()
            print(f"   Neo4j: {status}")

            for d in docs:
                print(f"\n📥 {d['filename']}")
                chunks = d["chunks"]
                if not args.no_qdrant:
                    try:
                        from ingestion.embedder import embed_chunks
                        chunks = embed_chunks(chunks)
                        load_chunks_to_qdrant(chunks)
                    except ImportError as e:
                        print(f"⚠️  Qdrant skip: {e} (install qdrant-client or use --no-qdrant)")

                counts = load_to_graph(
                    chunks             = chunks,
                    entities           = d["entities"],
                    relations          = d["relations"],
                    extraction_results = d["results"],
                    client             = neo4j,
                    filename           = d["filename"],
                    sha256             = d["sha256"],
                    stage              = "base_graph",
                )
                register_document(
                    sha256         = d["sha256"],
                    filename       = d["filename"],
                    client         = neo4j,
                    page_count     = len(d["pages"]),
                    entity_count   = counts["entities"],
                    relation_count = counts["relations"],
                )
                print(f"   Entities:  {counts['entities']}")
                print(f"   Relations: {counts['relations']}")
                print(f"   Chunks:    {counts['chunks']}")
                print(f"   Mentions:  {counts['mentions']}")

            neo4j.print_summary()
    except Exception as e:
        print(f"\n❌ Neo4j error: {e}")
        print("   Is Neo4j running? Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n🎉 Stage 1 complete in {elapsed:.1f}s")
    print("   Next: open Neo4j Browser → http://localhost:7474")
    print("   Verify: MATCH (n) WHERE n.stage='base_graph' RETURN labels(n), count(n)")
    print("   Then run Stage 2: python run_stage2.py --primekg data/seed/primekg_edges.csv")


if __name__ == "__main__":
    main()
