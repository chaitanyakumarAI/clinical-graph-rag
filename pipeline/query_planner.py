"""
pipeline/query_planner.py
Adaptive query planner — selects retrieval strategy based on question analysis.

Routes (4, not 3):
  vector_only    — factual/definition questions; no entity relationships needed
  hybrid_single  — one relationship hop + document retrieval
  hybrid_multi   — 2+ relationship hops + document retrieval
  gds_augmented  — graph algorithm reasoning (shortest path, similarity, centrality)

The route drives the full retrieval strategy in backend/main.py:
  vector_only   → Qdrant only; skip graph and GDS
  hybrid_single → Qdrant + single-hop Cypher (targeted by intent relation)
  hybrid_multi  → Qdrant + multi-hop Cypher + auto GDS shortest path (if 2 entities)
  gds_augmented → Qdrant + GDS algorithms as primary + Cypher as secondary

Phase 5: context_chunks from Qdrant retrieval enrich the entity identification.

Intent-Aware Enhancement:
  target_relations — Neo4j relation type(s) the question is asking about
  target_labels    — Node label type(s) the question is asking about
  These drive targeted Cypher in graph_retriever instead of blind expansion.
"""

from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, field_validator

from llm import RateLimitExhausted, chat_json

Route = Literal["vector_only", "hybrid_single", "hybrid_multi", "gds_augmented"]


class QueryPlan(BaseModel):
    route:               Route
    reason:              str
    entities_mentioned:  List[str] = []
    confidence:          float = 0.8
    retrieval_strategy:  str = ""
    # Intent-aware additions — drive targeted graph retrieval
    target_relations:    List[str] = []  # e.g. ["CAUSES", "HAS_SYMPTOM"]
    target_labels:       List[str] = []  # e.g. ["SideEffect", "Symptom"]

    @field_validator("confidence")
    @classmethod
    def clamp(cls, v):
        return max(0.0, min(1.0, float(v)))


SYSTEM_PROMPT = """You are a medical query router that selects the optimal retrieval strategy.

ROUTES — choose exactly one:

vector_only
  Use when: the question asks for definitions, mechanisms, descriptions, or general facts
  that can be answered from document text without traversing entity relationships.
  Examples: "What is metformin?", "How does insulin resistance develop?",
            "Describe the mechanism of EGFR signaling.", "What are clinical guidelines for hypertension?"

hybrid_single
  Use when: the answer requires ONE relationship traversal in the knowledge graph.
  The question names a specific entity and asks what is directly connected to it.
  Examples: "What drugs treat Type 2 Diabetes?", "What are the symptoms of dengue fever?",
            "What side effects does Metformin cause?", "Which organ does doxorubicin affect?"

hybrid_multi
  Use when: the answer requires TWO OR MORE relationship hops across different entity types.
  The question involves indirect or cross-domain connections.
  Examples: "Which diabetes drugs interact with hypertension medications?",
            "What diseases share symptoms with COVID but require different treatments?",
            "Which genes are targets of drugs that treat breast cancer?"

gds_augmented
  Use when: the question explicitly asks for path-finding, similarity, centrality, or
  ranking across the graph — things that need graph algorithm computation, not just traversal.
  Examples: "What connects Metformin to AMPK?",
            "Which entities are most central to insulin resistance?",
            "Find the shortest biological pathway between BRCA1 and breast cancer.",
            "What entities are most similar to Metformin in the knowledge graph?",
            "Which drug is most influential in the diabetes treatment network?"

CONFIDENCE: Rate your own routing confidence 0.0–1.0. Lower if question is ambiguous.

ENTITIES: List all medical entity names (drugs, diseases, genes, proteins, symptoms, pathways)
you identify in both the question and any provided document context.

TARGET_RELATIONS and TARGET_LABELS (critical for hybrid_single and hybrid_multi):
  Identify which Neo4j relationship type(s) the question is asking about.
  Use ONLY these allowed relation types:
    TREATS, CAUSES, INTERACTS_WITH, HAS_SYMPTOM, CONTRAINDICATED_FOR,
    AFFECTS_ORGAN, RECOMMENDED_BY, TARGETS, REGULATES, INHIBITS, ACTIVATES,
    LEADS_TO, BIOMARKER_FOR, EXPRESSED_IN, PARTICIPATES_IN, RELATED_TO

  Intent → target_relations mapping guide:
    "side effects / adverse effects / causes"         → ["CAUSES"]         labels: ["SideEffect"]
    "treats / medications for / drugs for"            → ["TREATS"]         labels: ["Drug","Treatment"]
    "symptoms of / signs of"                          → ["HAS_SYMPTOM"]    labels: ["Symptom"]
    "drug interactions / interacts with"              → ["INTERACTS_WITH"] labels: ["Drug"]
    "contraindicated / should not be used"            → ["CONTRAINDICATED_FOR"] labels: ["PatientGroup","Disease"]
    "affects organ / organ damage"                    → ["AFFECTS_ORGAN"]  labels: ["Organ"]
    "gene targets / targets protein"                  → ["TARGETS"]        labels: ["Gene","Protein"]
    "regulates / up-regulates / down-regulates"       → ["REGULATES","ACTIVATES","INHIBITS"] labels: ["Gene","Protein","Pathway"]
    "pathway / biological process"                    → ["PARTICIPATES_IN"] labels: ["Pathway","BiologicalProcess"]
    "biomarker for"                                   → ["BIOMARKER_FOR"]  labels: ["Disease"]
    "leads to / progression / complication"           → ["LEADS_TO"]       labels: ["Disease","Symptom"]
    "related to / associated with" (general)          → ["RELATED_TO"]     labels: []
    For vector_only or gds_augmented, set both to [].

Return ONLY valid JSON:
{
  "route": "...",
  "reason": "one-sentence explanation",
  "entities_mentioned": ["entity1", "entity2"],
  "confidence": 0.95,
  "retrieval_strategy": "short description of what retrieval will do",
  "target_relations": ["CAUSES"],
  "target_labels": ["SideEffect"]
}"""


def plan_query(
    query:          str,
    provider:       str = "auto",
    context_chunks: List[str] = [],
) -> QueryPlan:
    """
    Adaptively classify a user query into a retrieval route.

    Args:
        query:          The user's medical question
        provider:       "auto" | "gemini" | "groq" | "openai" | "ollama"
        context_chunks: Retrieved chunk snippets for entity grounding (Phase 5)

    Returns:
        QueryPlan with route, reason, entities, confidence, retrieval_strategy,
        target_relations, target_labels
    """
    user_content = f"Question: {query}"
    if context_chunks:
        snippet = "\n\n".join(c[:300] for c in context_chunks[:4])
        user_content += (
            f"\n\nRelevant document excerpts (use to identify entities):\n"
            f"---\n{snippet}\n---"
        )

    try:
        data = chat_json(SYSTEM_PROMPT, user_content, provider=provider, max_tokens=400)
        # Normalize route names for backward compatibility
        route = data.get("route", "hybrid_single")
        route = _normalize_route(route)
        data["route"] = route
        # Validate relation types against allowed list
        data["target_relations"] = _validate_relations(data.get("target_relations", []))
        return QueryPlan(**data)
    except RateLimitExhausted:
        raise   # no point continuing — the answer step would fail too
    except Exception as e:
        print(f"[query_planner] Planning failed ({e}), defaulting to hybrid_multi")
        return QueryPlan(
            route="hybrid_multi",
            reason="Classification failed — defaulting to hybrid_multi for safety",
            confidence=0.3,
        )


def _validate_relations(relations: list) -> List[str]:
    """Keep only known Neo4j relation types to prevent Cypher injection."""
    ALLOWED = {
        "TREATS", "CAUSES", "INTERACTS_WITH", "HAS_SYMPTOM",
        "CONTRAINDICATED_FOR", "AFFECTS_ORGAN", "RECOMMENDED_BY",
        "TARGETS", "REGULATES", "INHIBITS", "ACTIVATES",
        "LEADS_TO", "BIOMARKER_FOR", "EXPRESSED_IN",
        "PARTICIPATES_IN", "RELATED_TO",
    }
    valid = [r for r in (relations or []) if isinstance(r, str) and r.upper() in ALLOWED]
    return [r.upper() for r in valid]


def _normalize_route(route: str) -> str:
    """Map old route names to new adaptive route names."""
    mapping = {
        "no_graph":    "vector_only",
        "single_hop":  "hybrid_single",
        "multi_hop":   "hybrid_multi",
    }
    return mapping.get(route, route)


if __name__ == "__main__":
    tests = [
        ("What is metformin?",                                          "vector_only"),
        ("What are the common side effects of Metformin?",             "hybrid_single"),
        ("What drugs treat Type 2 Diabetes?",                          "hybrid_single"),
        ("Which diabetes drugs interact with hypertension medications?","hybrid_multi"),
        ("What connects Metformin to AMPK in the graph?",             "gds_augmented"),
        ("What is the shortest path from BRCA1 to chemotherapy?",     "gds_augmented"),
    ]
    print(f"{'Question':<60} {'Expected':<16} {'Got':<16} {'Relations':<20} Conf")
    print("-" * 120)
    for q, expected in tests:
        plan = plan_query(q)
        match = "✅" if plan.route == expected else "❌"
        rels = str(plan.target_relations)
        print(f"{match} {q[:57]:<57} {expected:<16} {plan.route:<16} {rels:<20} {plan.confidence:.2f}")
        print(f"   Labels: {plan.target_labels}  Strategy: {plan.retrieval_strategy}")
