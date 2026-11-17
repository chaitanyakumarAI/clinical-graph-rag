"""
evaluation/run_ragas.py
Evaluate the RAG pipeline with RAGAS, with and without query rewriting.

Steps (each one is cached per question, so an interrupted run resumes):
  1. answer  — run every question in testset.json through POST /query twice:
               "baseline" (rewriting off) and "rewrite" (rewriting on)
  2. judge   — score every answer with RAGAS metrics, using an LLM judge
  3. report  — python evaluation/make_report.py builds the PDF from the results

Usage (from the project root, Neo4j + Qdrant running, sample PDFs ingested):
    python evaluation/run_ragas.py            # answer + judge
    python evaluation/run_ragas.py answer     # only step 1
    python evaluation/run_ragas.py judge      # only step 2

    # Completeness check before/after on the multi-hop questions only,
    # written to a separate folder so the original results stay untouched:
    python evaluation/run_ragas.py --category multi-hop \\
        --configs rewrite completeness --out evaluation/results/fork
    python evaluation/compare.py evaluation/results/fork

Judge models are separate from the answering model (gpt-oss-120b). Groq's free
tier allows 200K tokens/day PER MODEL, and judging all metrics needs more than
that, so the metrics are split across two judge models (see METRIC_JUDGES).
Each metric always uses the same judge, so baseline vs rewrite stays a fair
comparison.
"""

from __future__ import annotations
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from llm import chat, usage_summary   # noqa: E402

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(EVAL_DIR, "results")
TESTSET_PATH = os.path.join(EVAL_DIR, "testset.json")

# metric → judge model. qwen3.8-27b is capped at ~1,000 output tokens/min on
# Groq's free tier, so its replies are limited by JUDGE_MAX_TOKENS below.
METRIC_JUDGES = {
    "faithfulness":       "openai/gpt-oss-20b",
    "answer_relevancy":   "openai/gpt-oss-20b",
    "context_recall":     "qwen/qwen3.8-27b",
    "answer_correctness": "qwen/qwen3.8-27b",
}
JUDGE_MAX_TOKENS = {"qwen/qwen3.8-27b": 900}
CONFIGS = {
    "baseline":     {"use_query_rewrite": False, "use_completeness_check": False},
    "rewrite":      {"use_query_rewrite": True,  "use_completeness_check": False},
    # Fork addition: rewrite + check that every part of the question is answered
    "completeness": {"use_query_rewrite": True,  "use_completeness_check": True},
}
DEFAULT_CONFIGS = ["baseline", "rewrite"]   # the original evaluation
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]


def _load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _record_usage(step):
    """Add this process's LLM token usage to results/usage.json under `step`."""
    path  = os.path.join(RESULTS_DIR, "usage.json")
    usage = _load_json(path, {})
    judge_models = set(METRIC_JUDGES.values())
    for model, counts in usage_summary().items():
        if (step == "judge") != (model in judge_models):
            continue   # count each model under the step that used it
        total = usage.setdefault(step, {}).setdefault(model, {"calls": 0, "tokens": 0})
        total["calls"]  += counts["calls"]
        total["tokens"] += counts["tokens"]
    _save_json(path, usage)


# ── Step 1: generate answers ──────────────────────────────────────────────────

def run_answers(testset, configs):
    from fastapi.testclient import TestClient
    import backend.main as backend

    client = TestClient(backend.app)
    for config in configs:
        options = CONFIGS[config]
        path    = os.path.join(RESULTS_DIR, f"answers_{config}.json")
        answers = _load_json(path, {})
        for item in testset:
            if item["id"] in answers:
                continue   # cached from an earlier run
            print(f"\n=== [{config}] {item['id']}: {item['question']}")
            t0 = time.time()
            response = client.post("/query", json={"question": item["question"], **options})
            if response.status_code == 429:
                print(f"⛔ Rate limit exhausted — re-run later to continue. {response.json()['detail']}")
                _record_usage("answer")
                return False
            response.raise_for_status()
            data = response.json()
            answers[item["id"]] = {
                "question":           item["question"],
                "rewritten_query":    data["rewritten_query"],
                "sub_queries":        data["sub_queries"],
                "answer":             data["answer"],
                "retrieved_contexts": data["retrieved_contexts"],
                "vector_sources":     data["vector_sources"],
                "route":              data["route"],
                "completeness":       data.get("completeness"),
                "latency_s":          round(time.time() - t0, 1),
            }
            _save_json(path, answers)
    _record_usage("answer")
    print("\nAnswer step done. LLM usage:", usage_summary())
    return True


# ── Step 2: judge with RAGAS ──────────────────────────────────────────────────

def _make_judge(model: str):
    """A RAGAS LLM that routes every judge call through llm/client.py (rate limited)."""
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    @dataclass
    class RateLimitedJudge(BaseRagasLLM):
        def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            texts = [
                chat("You are a meticulous evaluator. Follow the requested output format exactly.",
                     prompt.to_string(), provider="groq", model=model,
                     max_tokens=JUDGE_MAX_TOKENS.get(model, 2048),
                     temperature=temperature or 0.0)
                for _ in range(n)
            ]
            return LLMResult(generations=[[Generation(text=t) for t in texts]])

        async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

        def is_finished(self, response):
            return True

    return RateLimitedJudge()


def _make_metrics():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import AnswerCorrectness, Faithfulness, LLMContextRecall, ResponseRelevancy

    # Local embeddings (no API quota) — same model the RAG uses for chunks.
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))
    judge = {name: _make_judge(model) for name, model in METRIC_JUDGES.items()}
    return [
        Faithfulness(llm=judge["faithfulness"]),
        ResponseRelevancy(llm=judge["answer_relevancy"], embeddings=embeddings, strictness=1),
        LLMContextRecall(llm=judge["context_recall"]),
        AnswerCorrectness(llm=judge["answer_correctness"], embeddings=embeddings),
    ]


def run_judge(testset, configs):
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    metrics    = _make_metrics()
    run_config = RunConfig(max_workers=1, timeout=900, max_retries=2)
    references = {item["id"]: item["reference"] for item in testset}

    for config in configs:
        answers = _load_json(os.path.join(RESULTS_DIR, f"answers_{config}.json"), {})
        path    = os.path.join(RESULTS_DIR, f"scores_{config}.json")
        scores  = _load_json(path, {})
        for qid, ans in answers.items():
            if qid not in references:
                continue   # outside the --ids / --category selection
            if qid in scores and not any(math.isnan(scores[qid].get(m, float("nan"))) for m in METRIC_NAMES):
                continue   # already scored
            print(f"\n=== judging [{config}] {qid}", flush=True)
            dataset = EvaluationDataset.from_list([{
                "user_input":         ans["question"],
                "response":           ans["answer"],
                "retrieved_contexts": ans["retrieved_contexts"],
                "reference":          references[qid],
            }])
            # raise_exceptions=False: if a judge reply can't be parsed (or the
            # quota runs out) only that metric becomes NaN; it is retried on the
            # next run instead of aborting everything.
            result = evaluate(dataset, metrics=metrics, run_config=run_config,
                              show_progress=False, raise_exceptions=False)
            row = result.to_pandas().iloc[0]
            scores[qid] = {m: float(row[m]) for m in METRIC_NAMES}
            print("   ", {m: round(v, 3) for m, v in scores[qid].items()}, flush=True)
            _save_json(path, scores)

    missing = [(c, q) for c in configs for q, s in
               _load_json(os.path.join(RESULTS_DIR, f"scores_{c}.json"), {}).items()
               if q in references and any(math.isnan(v) for v in s.values())]
    print(f"\nJudge step done. Missing scores (re-run to retry): {missing or 'none'}")
    print("LLM usage:", usage_summary())
    return not missing


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument("step", nargs="?", default="all", choices=["all", "answer", "judge"])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list(CONFIGS),
                        help="pipeline configurations to run (default: baseline rewrite)")
    parser.add_argument("--category", nargs="+",
                        help="only questions of these categories, e.g. multi-hop")
    parser.add_argument("--ids", nargs="+", help="only these question ids, e.g. Q8 Q9")
    parser.add_argument("--out", default=RESULTS_DIR,
                        help="results directory (default: evaluation/results)")
    args = parser.parse_args()

    RESULTS_DIR = os.path.abspath(args.out)   # read by every helper above
    testset = [item for item in _load_json(TESTSET_PATH, [])
               if (not args.category or item["category"] in args.category)
               and (not args.ids or item["id"] in args.ids)]
    print(f"{len(testset)} question(s), configs={args.configs}, results → {RESULTS_DIR}")

    ok = True
    if args.step in ("all", "answer"):
        ok = run_answers(testset, args.configs)
    if ok and args.step in ("all", "judge"):
        try:
            ok = run_judge(testset, args.configs)
        finally:
            _record_usage("judge")
    sys.exit(0 if ok else 2)
