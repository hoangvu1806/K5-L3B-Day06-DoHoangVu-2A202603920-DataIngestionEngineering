"""Score retrieved evidence and aggregate Hit@1 / Evidence Hit@3.

A retrieved chunk is a HIT when (GUIDE.md §4):
  1. it comes from a gold page: its page list intersects question.gold_pages
     (and its document is gold_source_id), AND
  2. it contains the WHOLE evidence_quote (compared after collapsing whitespace).
Hit@1 = top-1 is a hit; Evidence Hit@3 = at least one hit in the top 3.

These are automatic retrieval metrics only. Chunk coherence and grounding are
human labels (results/audit_*.csv) and are never inferred from string match.
"""
from __future__ import annotations

import re
from typing import Any


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Label each retrieved chunk and compute hit@1 / hit@3 for one question."""
    evidence = squash(question["evidence_quote"])
    gold_pages = set(question["gold_pages"])
    labelled = []
    for rank, h in enumerate(hits, start=1):
        md = h["metadata"]
        page_match = (md.get("document_id") == question["gold_source_id"]
                      and bool(gold_pages & set(md.get("pages", [md.get("page")]))))
        evidence_in = evidence in squash(h["text"])
        labelled.append({
            "rank": rank, "chunk_id": h["chunk_id"], "score": round(h["score"], 4),
            "page": md.get("page"), "pages": md.get("pages"),
            "disease_no": md.get("disease_no"), "disease": md.get("disease"),
            "section": md.get("section"), "subsection": md.get("subsection"),
            "special_id": md.get("special_id"), "source": md.get("source"),
            "page_match": page_match, "evidence_in_chunk": evidence_in,
            "hit": page_match and evidence_in, "text": h["text"],
        })
    flt = question.get("filter")
    compliant = None
    if flt:
        compliant = all(all(l.get(k) == v for k, v in flt.items()) for l in labelled)
    return {
        "hit1": bool(labelled) and labelled[0]["hit"],
        "hit3": any(l["hit"] for l in labelled[:3]),
        "page_hit3": any(l["page_match"] for l in labelled[:3]),
        "first_hit_rank": next((l["rank"] for l in labelled if l["hit"]), None),
        "filter_compliant": compliant,
        "top": labelled,
    }


def _ratio(n: int, d: int) -> dict[str, Any]:
    return {"n": n, "N": d, "rate": round(n / d, 3) if d else None, "text": f"{n}/{d}"}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by strategy and split. Reports counts (n/N), not only rates."""
    out: dict[str, Any] = {}
    for t in traces:
        out.setdefault(t["strategy"], {}).setdefault(t["split"], []).append(t)
    summary: dict[str, Any] = {}
    for strategy, by_split in sorted(out.items()):
        summary[strategy] = {}
        for split, rows in sorted(by_split.items()):
            filtered = [r for r in rows if r["filter_applied"]]
            with_filter_q = [r for r in rows if r["question_filter"]]
            summary[strategy][split] = {
                "hit1": _ratio(sum(r["hit1"] for r in rows), len(rows)),
                "evidence_hit3": _ratio(sum(r["hit3"] for r in rows), len(rows)),
                "page_hit3": _ratio(sum(r["page_hit3"] for r in rows), len(rows)),
                "filter_questions_hit3": _ratio(sum(r["hit3"] for r in with_filter_q), len(with_filter_q)),
                "filter_compliance": (_ratio(sum(bool(r["filter_compliant"]) for r in filtered), len(filtered))
                                      if filtered else "N/A (no filter applied)"),
                "misses_hit3": [r["id"] for r in rows if not r["hit3"]],
                "misses_hit1": [r["id"] for r in rows if not r["hit1"]],
            }
    return summary


def select_strategy(dev_summary: dict[str, Any], chunk_counts: dict[str, int]) -> dict[str, Any]:
    """Apply artifacts/selection_criteria.md to DEV results only.

    1. highest dev Evidence Hit@3; 2. if within 1 question, higher dev Hit@1;
    3. still tied: fewer chunks.
    """
    rows = {s: (v["dev"]["evidence_hit3"]["n"], v["dev"]["hit1"]["n"], -chunk_counts[s])
            for s, v in dev_summary.items()}
    best_h3 = max(r[0] for r in rows.values())
    contenders = [s for s, r in rows.items() if best_h3 - r[0] <= 1]
    steps = [f"Rule 1: dev Evidence Hit@3 = {{{', '.join(f'{s}: {rows[s][0]}' for s in sorted(rows))}}}; "
             f"within 1 of best ({best_h3}): {sorted(contenders)}"]
    if len(contenders) > 1:
        best_h1 = max(rows[s][1] for s in contenders)
        contenders2 = [s for s in contenders if rows[s][1] == best_h1]
        steps.append(f"Rule 2: dev Hit@1 among them = {{{', '.join(f'{s}: {rows[s][1]}' for s in sorted(contenders))}}}; "
                     f"best: {sorted(contenders2)}")
        contenders = contenders2
    if len(contenders) > 1:
        fewest = max(rows[s][2] for s in contenders)
        contenders3 = [s for s in contenders if rows[s][2] == fewest]
        steps.append(f"Rule 3: chunks = {{{', '.join(f'{s}: {-rows[s][2]}' for s in sorted(contenders))}}}; "
                     f"fewest: {sorted(contenders3)}")
        contenders = contenders3
    return {"chosen": sorted(contenders), "steps": steps}
