"""
run_stage2.py — Stage 2 CLI entry point.

Usage:
    python run_stage2.py --nodes  data/seed/primekg_nodes.csv \\
                         --edges  data/seed/primekg_edges.csv

    # Save match report without writing to Neo4j:
    python run_stage2.py --nodes data/seed/primekg_nodes.csv \\
                         --edges data/seed/primekg_edges.csv \\
                         --dry-run --output-json data/seed/match_report.json

PrimeKG download:
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
    Files needed: kg.csv (edges), nodes.csv (node index)
"""

from __future__ import annotations
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from graph.neo4j_client               import Neo4jClient
from enrichment.primekg_matcher       import PrimeKGMatcher, load_neo4j_entities
from enrichment.primekg_edge_filter   import filter_edges, build_match_maps
from enrichment.primekg_loader        import write_primekg_ids, load_enriched_edges


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: Selective PrimeKG enrichment of the base graph"
    )
    parser.add_argument("--nodes", required=True, help="PrimeKG nodes CSV (node_index, node_id, node_type, node_name)")
    parser.add_argument("--edges", required=True, help="PrimeKG edges CSV (relation, x_id, y_id, ...)")
    parser.add_argument("--dry-run",     action="store_true", help="Match only, don't write to Neo4j")
    parser.add_argument("--output-json", type=str, default=None, help="Save match report to JSON")
    parser.add_argument("--threshold",   type=int, default=88, help="Fuzzy match threshold (default: 88)")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    print("=" * 65)
    print("  Stage 2: PrimeKG Enrichment (Selective)")
    print("=" * 65)

    # Validate inputs
    for path in [args.nodes, args.edges]:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            sys.exit(1)

    os.environ["PRIMEKG_FUZZY_THRESHOLD"] = str(args.threshold)

    # ── Step 1: Fetch Stage 1 entities from Neo4j ─────────────────────────
    print("\n[1/4] Fetching Stage 1 entities from Neo4j …")
    try:
        with Neo4jClient() as neo4j:
            neo4j_entities = load_neo4j_entities(neo4j)
    except Exception as e:
        print(f"❌ Neo4j error: {e}")
        sys.exit(1)

    if not neo4j_entities:
        print("❌ No Stage 1 entities found. Run Stage 1 first.")
        sys.exit(1)

    print(f"   Found {len(neo4j_entities)} entities (stage='base_graph')")

    # ── Step 2: Match to PrimeKG ──────────────────────────────────────────
    print("\n[2/4] Matching entities to PrimeKG …")
    matcher      = PrimeKGMatcher(args.nodes)
    match_results = matcher.match_all(neo4j_entities)

    matched   = [m for m in match_results if m.get("matched")]
    unmatched = [m for m in match_results if not m.get("matched")]
    print(f"   Matched: {len(matched)} | Unmatched: {len(unmatched)}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({
                "matched":   matched,
                "unmatched": [m["neo4j_name"] for m in unmatched],
            }, f, indent=2)
        print(f"   Match report saved: {args.output_json}")

    if args.dry_run:
        print("\n🔍 Dry run — skipping Neo4j write.")
        elapsed = time.time() - start
        print(f"⏱️  Stage 2 dry run done in {elapsed:.1f}s")
        return

    # ── Step 3: Filter PrimeKG edges ─────────────────────────────────────
    print("\n[3/4] Filtering PrimeKG edges (both endpoints must be matched) …")
    matched_ids, id_to_neo4j = build_match_maps(match_results)
    filtered_edges = filter_edges(args.edges, matched_ids, id_to_neo4j)

    if not filtered_edges:
        print("⚠️  No edges passed the filter. "
              "This may mean your extracted entities are too few or too specific.")
        elapsed = time.time() - start
        print(f"\n⏱️  Stage 2 done in {elapsed:.1f}s (no edges imported)")
        return

    # ── Step 4: Write to Neo4j ────────────────────────────────────────────
    print(f"\n[4/4] Writing {len(filtered_edges):,} enriched edges to Neo4j …")
    try:
        with Neo4jClient() as neo4j:
            write_primekg_ids(match_results, neo4j)
            load_enriched_edges(filtered_edges, neo4j)
            neo4j.print_summary()
    except Exception as e:
        print(f"❌ Neo4j error: {e}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n🎉 Stage 2 complete in {elapsed:.1f}s")
    print("   Verification:")
    print("   MATCH ()-[r {stage:'primekg_enriched'}]->() RETURN count(r)")
    print("   MATCH (n) WHERE n.primekg_id IS NOT NULL RETURN labels(n), count(n)")
    print("\n   Next: python backend/main.py  (Stage 3 — Hybrid Retrieval)")


if __name__ == "__main__":
    main()
