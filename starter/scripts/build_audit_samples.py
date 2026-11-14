"""Extract the fixed manual-audit samples (no labels) from chunks and traces.

Coherence: the first line of section 'IV. ĐIỀU TRỊ' in diseases 1-10 (fixed
before looking at any label). For each strategy, the audited chunk is the first
chunk (by chunk order) whose text contains that heading line plus the next 40
characters, i.e. the chunk holding that source position. C uses B's chunks.
Metadata / grounding: every test question x top-3 x strategy from traces_test.

Run from starter/:  python scripts/build_audit_samples.py
Writes results/audit_samples.json (input for labelling).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "starter" / "results"
squash = lambda t: re.sub(r"\s+", " ", t).strip()

pages = {json.loads(l)["page"]: json.loads(l) for l in (R / "pages_normalized.jsonl").read_text(encoding="utf-8").splitlines()}
chunks = {s: [json.loads(l) for l in (R / f"chunks_{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in "AB"}

positions = []
for d in range(1, 11):
    for p in sorted(pages):
        md = pages[p]["metadata"]
        h = next((h for h in md["headings"] if md["disease_no"] == d and h["level"] == "section"
                  and h["text"].startswith("IV.")), None)
        if h:
            anchor = squash(pages[p]["text"][h["offset"]:h["offset"] + 60])
            positions.append({"pos_id": f"cp{d:02d}", "disease_no": d, "pdf_page": p,
                              "heading": h["text"], "anchor": anchor})
            break

coherence = []
for pos in positions:
    for s in "ABC":
        src = chunks["B" if s == "C" else s]
        c = next((c for c in src if pos["disease_no"] == c["metadata"]["disease_no"]
                  and pos["anchor"][:40] in squash(c["text"])), None)
        if c is None:  # A may cut the 40-char anchor; fall back to the heading line alone
            c = next(c for c in src if pos["disease_no"] == c["metadata"]["disease_no"]
                     and pos["pdf_page"] in c["metadata"]["pages"] and pos["heading"] in c["text"])
        coherence.append({**pos, "strategy": s, "chunk_id": c["chunk_id"], "pages": c["metadata"]["pages"],
                          "chars": len(c["text"]), "text": c["text"]})

traces = [json.loads(l) for l in (R / "traces_test.jsonl").read_text(encoding="utf-8").splitlines()]
retrieval = [{"id": t["id"], "strategy": t["strategy"], "question": t["question"],
              "expected_answer": t["expected_answer"], "gold_pages": t["gold_pages"],
              "evidence_quote": t["evidence_quote"], "filter": t["question_filter"],
              "auto_hit3": t["hit3"], "top": t["top"]} for t in traces]

(R / "audit_samples.json").write_text(json.dumps({"coherence": coherence, "retrieval_test": retrieval},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
print(len(positions), "positions;", len(coherence), "coherence rows;", len(retrieval), "retrieval rows")
for p in positions: print(p["pos_id"], p["pdf_page"], p["anchor"][:50])
