"""Aggregate audit_*.csv into results/audit_summary.json for the report."""
import csv
import json
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def coherence():
    by = defaultdict(list)
    for row in csv.DictReader(open(R / "audit_coherence.csv", encoding="utf-8")):
        by[row["strategy"]].append(int(row["label"]))
    return {s: {"mean": mean(v), "n": len(v), "dist": {k: v.count(k) for k in (0, 1, 2)}}
            for s, v in sorted(by.items())}


def grounding():
    by = defaultdict(list)
    for row in csv.DictReader(open(R / "audit_grounding.csv", encoding="utf-8")):
        by[row["strategy"]].append(int(row["label"]))
    return {s: {"mean": mean(v), "n": len(v), "dist": {k: v.count(k) for k in (0, 1, 2)}}
            for s, v in sorted(by.items())}


def metadata():
    by = defaultdict(lambda: {"correct": 0, "total": 0, "filter_compliant_num": 0, "filter_compliant_den": 0})
    for row in csv.DictReader(open(R / "audit_metadata.csv", encoding="utf-8")):
        s = row["strategy"]
        by[s]["correct"] += int(row["correct_records"])
        by[s]["total"] += int(row["total_records"])
        fc = row["filter_compliance"]
        if "/" in fc and "N/A" not in fc:
            n, d = fc.split("/")
            by[s]["filter_compliant_num"] += int(n)
            by[s]["filter_compliant_den"] += int(d)
    out = {}
    for s, v in sorted(by.items()):
        out[s] = {"correct_records": f"{v['correct']}/{v['total']}",
                  "filter_compliance": (f"{v['filter_compliant_num']}/{v['filter_compliant_den']}"
                                        if v["filter_compliant_den"] else "N/A (no filter applied)")}
    return out


summary = {"coherence": coherence(), "metadata": metadata(), "grounding": grounding(),
           "method": "Nhãn do tác giả chấm, 10 vị trí coherence (dòng IV. ĐIỀU TRỊ của bệnh 1-10) chọn cố định trước khi xem điểm; grounding và metadata trên 6 câu test × top-3 × 3 cấu hình."}
(R / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
