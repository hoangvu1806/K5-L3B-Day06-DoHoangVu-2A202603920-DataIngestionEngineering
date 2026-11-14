"""Read the frozen Day 06 benchmark results for the UI's Benchmark tab.

Everything comes from files already produced by the pipeline (no new runs):
  benchmark/questions.json                 questions, gold pages, evidence, filters
  starter/results/traces_{dev,test}.jsonl  top-3 chunks per question x strategy
  starter/results/metrics_{dev,test}.json  Hit@1 / Evidence Hit@3 / filter compliance
  starter/results/audit_summary.json       manual coherence / metadata / grounding
  starter/results/audit_grounding.csv      per-question grounding labels (test)
  starter/results/chunk_stats.json         chunk counts and length stats
  artifacts/selection_decision.json        the dev-only selection of a strategy
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "starter" / "results"

STRATEGIES = {
    "A": "Cắt cố định 500 ký tự, overlap 50, không lọc",
    "B": "Cắt theo mục/tiểu mục (tối đa 1000 ký tự), không lọc",
    "C": "Giống B + lọc disease_no khi câu hỏi có filter",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=1)
def load_benchmark() -> dict[str, Any]:
    questions = _json(ROOT / "benchmark" / "questions.json")["questions"]
    metrics = {s: _json(RESULTS / f"metrics_{s}.json") for s in ("dev", "test")}
    audit = _json(RESULTS / "audit_summary.json")
    chunk_stats = _json(RESULTS / "chunk_stats.json")["strategies"]
    decision = _json(ROOT / "artifacts" / "selection_decision.json")
    grounding = {(r["question_id"], r["strategy"]): {"label": int(r["label"]), "reason": r["reason"]}
                 for r in csv.DictReader((RESULTS / "audit_grounding.csv").open(encoding="utf-8"))}

    traces: dict[tuple[str, str], dict] = {}
    for split in ("dev", "test"):
        for t in _jsonl(RESULTS / f"traces_{split}.jsonl"):
            traces[(t["id"], t["strategy"])] = t

    chosen = decision.get("chosen", [])
    strategies = {}
    for s, desc in STRATEGIES.items():
        cs = chunk_stats.get(s if s != "C" else "B", {})
        strategies[s] = {
            "desc": desc,
            "selected": s in chosen,
            "chunks": cs.get("chunks"),
            "len_mean": cs.get("len_mean"),
            "dev": metrics["dev"]["summary"][s],
            "test": metrics["test"]["summary"][s],
            "coherence": audit["coherence"].get(s),
            "grounding": audit["grounding"].get(s),
            "metadata": audit["metadata"].get(s),
        }

    rows = []
    for q in questions:
        results = {}
        for s in STRATEGIES:
            t = traces.get((q["id"], s))
            if not t:
                continue
            results[s] = {
                "hit1": t["hit1"], "hit3": t["hit3"], "page_hit3": t["page_hit3"],
                "first_hit_rank": t["first_hit_rank"], "filter_applied": t["filter_applied"],
                "filter_compliant": t["filter_compliant"],
                "grounding": grounding.get((q["id"], s)),
                "top": [{k: h.get(k) for k in ("rank", "score", "pages", "disease_no", "disease", "section",
                                               "subsection", "special_id", "page_match",
                                               "evidence_in_chunk", "hit", "text")} for h in t["top"]],
            }
        rows.append({
            "id": q["id"], "split": q["split"], "type": q["type"], "disease_no": q.get("disease_no"),
            "question": q["question"], "expected_answer": q["expected_answer"],
            "gold_pages": q["gold_pages"], "evidence_quote": q["evidence_quote"],
            "filter": q.get("filter"), "filter_rationale": q.get("filter_rationale"),
            "gold_note": q.get("gold_note"), "results": results,
        })

    return {
        "strategies": strategies,
        "selection": {"chosen": chosen, "steps": decision.get("steps", []),
                      "criteria": "artifacts/selection_criteria.md"},
        "run": {s: {"run_at": metrics[s]["run_at"], "n": metrics[s]["n_questions"],
                    "top_k": metrics[s]["top_k"], "model": metrics[s]["model"]} for s in ("dev", "test")},
        "audit_method": audit.get("method"),
        "questions": rows,
    }
