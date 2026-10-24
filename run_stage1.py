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

# Make imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

import openai

from ingestion.pdf_parser    import parse_pdf, parse_pdf_directory
from ingestion.chunker       import chunk_pages
from ingestion.entity_extractor import extract_from_chunks
from ingestion.deduplicator  import deduplicate
from graph.neo4j_client      import Neo4jClient
from graph.graph_loader      import load_to_graph, load_chunks_to_qdrant
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


def main():
    args = parse_args()
    start = time.time()

    print("=" * 65)
    print("  Stage 1: Base Graph — PDF → Neo4j")
    print("=" * 65)

    # ── Step 1: Parse ──────────────────────────────────────────────────────
    print("\n[1/5] Parsing PDF(s) …")
    if args.pdf:
        pages = parse_pdf(args.pdf)
    else:
        pages = parse_pdf_directory(args.pdf_dir)

    if not pages:
        print("❌ No pages extracted. Check your PDF path.")
        sys.exit(1)

    # ── Step 2: Chunk ──────────────────────────────────────────────────────
    print("\n[2/5] Chunking …")
    chunks = chunk_pages(pages, args.chunk_tokens, args.overlap_tokens)

    # ── Step 3: LLM Entity Extraction ─────────────────────────────────────
    provider = resolve_provider(args.provider)
    print(f"\n[3/5] Extracting entities via LLM ({provider}: {model_for(provider)}) …")
    try:
        results = extract_from_chunks(
            chunks,
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

    # ── Step 4: Deduplicate ────────────────────────────────────────────────
    print("\n[4/5] Deduplicating …")
    entities, relations = deduplicate(results)

    # ── Step 5: Embed + Store in Qdrant ───────────────────────────────────
    if not args.no_qdrant and not args.dry_run:
        print("\n[5/5] Embedding chunks → Qdrant …")
        try:
            from ingestion.embedder import embed_chunks
            chunks = embed_chunks(chunks)
            load_chunks_to_qdrant(chunks)
        except ImportError as e:
            print(f"⚠️  Qdrant skip: {e} (install qdrant-client or use --no-qdrant)")
    else:
        print("\n[5/5] Skipping Qdrant (--no-qdrant or --dry-run)")

    # ── Optional: save JSON for inspection ────────────────────────────────
    if args.output_json:
        out = {"entities": entities, "relations": relations}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n📄 Saved extraction to: {args.output_json}")

    if args.dry_run:
        print("\n🔍 Dry run — skipping Neo4j write.")
        print(f"   Entities extracted: {len(entities)}")
        print(f"   Relations extracted: {len(relations)}")
        elapsed = time.time() - start
        print(f"\n⏱️  Total time: {elapsed:.1f}s")
        return

    # ── Load to Neo4j ──────────────────────────────────────────────────────
    print("\n📥 Loading to Neo4j …")
    try:
        with Neo4jClient() as neo4j:
            status = neo4j.verify_connection()
            print(f"   Neo4j: {status}")

            counts = load_to_graph(
                chunks             = chunks,
                entities           = entities,
                relations          = relations,
                extraction_results = results,
                client             = neo4j,
                stage              = "base_graph",
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
