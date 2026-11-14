"""Run the benchmark: top-3 retrieval for every question x strategy A/B/C.

Protocol (artifacts/selection_criteria.md):
  python -m src.run --split dev       # A, B, C on the 18 dev questions
  python -m src.run --select          # apply the frozen rule to DEV results,
                                      # write artifacts/selection_decision.json
  python -m src.run --split test      # ONCE, only after a decision exists

Guards: the test split refuses to run without a recorded decision, and refuses
to overwrite existing test traces (no silent re-runs after seeing test).

Prerequisites: python -m src.ingestion, python -m src.chunking, python -m src.index.
Outputs: results/traces_<split>.jsonl, results/metrics_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .embeddings import OpenRouterEmbedder
from .evaluation import evaluate_query, select_strategy, summarize
from .store import QdrantStore

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "starter" / "results"
QUESTIONS = ROOT / "benchmark" / "questions.json"
DECISION = ROOT / "artifacts" / "selection_decision.json"
STRATEGIES = ("A", "B", "C")
TOP_K = 3


def run_split(split: str) -> None:
    bench = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = [q for q in bench["questions"] if q["split"] == split]
    load_dotenv(ROOT / "starter" / ".env")
    embedder = OpenRouterEmbedder()
    store = QdrantStore(os.environ["QDRANT_URL"], os.environ.get("QDRANT_API_KEY"), embedder.dimensions)

    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cost0 = embedder.cost_usd
    t0 = time.perf_counter()
    qvecs = {q["id"]: embedder.embed_query(q["question"]) for q in questions}  # shared by A/B/C
    embed_s = time.perf_counter() - t0

    traces = []
    for strategy in STRATEGIES:
        for q in questions:
            flt = q["filter"] if strategy == "C" and q["filter"] else None
            t = time.perf_counter()
            hits = store.search(strategy, qvecs[q["id"]], TOP_K, flt)
            latency = time.perf_counter() - t
            res = evaluate_query(q, hits)
            traces.append({
                "run_at": run_at, "strategy": strategy, "collection": store.collection(strategy),
                "split": split, "id": q["id"], "type": q["type"], "question": q["question"],
                "expected_answer": q["expected_answer"], "gold_source_id": q["gold_source_id"],
                "gold_pages": q["gold_pages"], "evidence_quote": q["evidence_quote"],
                "question_filter": q["filter"], "filter_applied": flt,
                "search_ms": round(latency * 1000, 1), **res,
            })

    metrics = {
        "run_at": run_at, "split": split, "n_questions": len(questions), "top_k": TOP_K,
        "model": embedder.model_name, "dimensions": embedder.dimensions,
        "query_embedding_cost_usd": round(embedder.cost_usd - cost0, 6),
        "query_embedding_seconds": round(embed_s, 1),
        "summary": {s: v[split] for s, v in summarize(traces).items()},
    }
    with (RESULTS / f"traces_{split}.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    (RESULTS / f"metrics_{split}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(f"split={split} questions={len(questions)} query-embedding cost=${metrics['query_embedding_cost_usd']}")
    print(f"{'strategy':8} {'Hit@1':>7} {'EvHit@3':>8} {'PageHit@3':>10} {'filterQ Hit@3':>14} {'filter comply':>14}  misses@3")
    for s, v in metrics["summary"].items():
        fc = v["filter_compliance"] if isinstance(v["filter_compliance"], str) else v["filter_compliance"]["text"]
        print(f"{s:8} {v['hit1']['text']:>7} {v['evidence_hit3']['text']:>8} {v['page_hit3']['text']:>10} "
              f"{v['filter_questions_hit3']['text']:>14} {fc:>14}  {v['misses_hit3']}")


def select() -> None:
    dev = json.loads((RESULTS / "metrics_dev.json").read_text(encoding="utf-8"))
    stats = json.loads((RESULTS / "chunk_stats.json").read_text(encoding="utf-8"))["strategies"]
    counts = {"A": stats["A"]["chunks"], "B": stats["B"]["chunks"], "C": stats["B"]["chunks"]}
    decision = select_strategy({s: {"dev": v} for s, v in dev["summary"].items()}, counts)
    decision.update({"decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "based_on": "results/metrics_dev.json (dev split only)",
                     "criteria": "artifacts/selection_criteria.md"})
    if len(decision["chosen"]) != 1:
        decision["status"] = "tie: the frozen criteria do not separate these; author must decide and record why"
    else:
        decision["status"] = "decided"
    DECISION.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    if args.select:
        select()
        return
    if args.split == "test":
        if not DECISION.exists() or json.loads(DECISION.read_text(encoding="utf-8")).get("status") != "decided":
            raise SystemExit("Refusing to run test: record a dev-based decision first (python -m src.run --select).")
        if (RESULTS / "traces_test.jsonl").exists():
            raise SystemExit("Refusing to overwrite test traces: the test split runs once.")
    if args.split:
        run_split(args.split)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
