"""Thin RAG wrapper around the Day 06 pipeline.

Reuses the pipeline's embedding provider and Qdrant client (strategy B / C):
  - retrieve top-k chunks from collection `day06_rhm_b`, with optional
    `disease_no` filter supplied by the UI (this is the Day 06 'strategy C');
  - ask the chat model for a grounded answer, told to cite [#n] by chunk index
    and to say 'Không rõ' when the context is insufficient.

Uses OpenRouter for both embeddings and chat (the user's existing key).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

# Reuse the Day 06 pipeline modules without copying code.
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from src.embeddings import OpenRouterEmbedder  # noqa: E402
from src.store import QdrantStore              # noqa: E402

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_STRATEGY = "C"      # B's chunks + optional disease filter
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = (
    "Bạn là trợ lý tra cứu tài liệu chuyên môn 'Hướng dẫn chẩn đoán và điều trị "
    "một số bệnh về Răng Hàm Mặt' do Bộ Y tế ban hành kèm Quyết định 3108/QĐ-BYT "
    "ngày 28/7/2015. Trả lời ngắn gọn, có cấu trúc, bằng tiếng Việt.\n\n"
    "Quy tắc:\n"
    "1. Chỉ dùng thông tin trong các đoạn tài liệu được cung cấp. KHÔNG tự suy luận "
    "ngoài tài liệu.\n"
    "2. Trích nguồn bằng ký hiệu [#n] (n là số thứ tự đoạn), ví dụ: 'Nên nhổ sớm [#1].'\n"
    "3. Nếu tài liệu không có thông tin cần thiết, trả lời 'Không rõ, tài liệu "
    "không nêu.' và giải thích ngắn.\n"
    "4. Không đưa chẩn đoán cá nhân hoặc chỉ định điều trị cho người cụ thể; "
    "đây là công cụ tra cứu tài liệu, không thay thế bác sĩ.\n"
)


def load_env(path: Path) -> None:
    """Read 'KEY=VALUE' or 'export KEY=VALUE'; do not overwrite existing env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep and key.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env(ROOT / "starter" / ".env")


class RagEngine:
    def __init__(self) -> None:
        self.embedder = OpenRouterEmbedder()
        self.store = QdrantStore(os.environ["QDRANT_URL"],
                                 os.environ.get("QDRANT_API_KEY"),
                                 self.embedder.dimensions)
        self.chat_model = os.getenv("CHAT_MODEL", "deepseek/deepseek-v4.1-flash")
        self.api_key = os.environ["OPENROUTER_API_KEY"]

    def retrieve(self, question: str, disease_no: int | None, top_k: int) -> list[dict]:
        strategy = "C" if disease_no else "B"
        filters = {"disease_no": disease_no} if disease_no else None
        vec = self.embedder.embed_query(question)
        hits = self.store.search(strategy, vec, top_k, filters)
        for i, h in enumerate(hits, 1):
            h["citation"] = f"#{i}"
        return hits

    def _format_context(self, hits: list[dict]) -> str:
        parts = []
        for h in hits:
            md = h["metadata"]
            label = []
            if md.get("disease"):
                label.append(f"Bệnh {md.get('disease_no')}. {md['disease']}")
            elif md.get("special_id"):
                label.append(md.get("section") or md["special_id"])
            if md.get("section"):
                label.append(md["section"])
            if md.get("subsection"):
                label.append(md["subsection"])
            pages = md.get("pages") or [md.get("page")]
            parts.append(f"[{h['citation']}] ({' > '.join(label) or '—'}; trang {pages})\n{h['text']}")
        return "\n\n".join(parts)

    def answer(self, question: str, disease_no: int | None = None,
               top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        t0 = time.perf_counter()
        hits = self.retrieve(question, disease_no, top_k)
        context = self._format_context(hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Câu hỏi: {question}\n\nCác đoạn tài liệu tham khảo:\n\n{context}\n\n"
                "Dựa trên các đoạn trên, hãy trả lời câu hỏi, trích nguồn [#n]."},
        ]
        payload = {"model": self.chat_model, "messages": messages, "temperature": 0.2}
        resp = requests.post(CHAT_URL, headers={"Authorization": f"Bearer {self.api_key}"},
                             json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        answer_text = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return {
            "answer": answer_text,
            "sources": [{
                "citation": h["citation"],
                "chunk_id": h["chunk_id"],
                "score": round(h["score"], 3),
                "disease_no": h["metadata"].get("disease_no"),
                "disease": h["metadata"].get("disease"),
                "section": h["metadata"].get("section"),
                "subsection": h["metadata"].get("subsection"),
                "special_id": h["metadata"].get("special_id"),
                "pages": h["metadata"].get("pages") or [h["metadata"].get("page")],
                "text": h["text"],
            } for h in hits],
            "strategy": "C" if disease_no else "B",
            "disease_no": disease_no,
            "top_k": top_k,
            "model": self.chat_model,
            "embedding_model": self.embedder.model_name,
            "tokens": usage.get("total_tokens"),
            "cost_usd": usage.get("cost"),
            "seconds": round(time.perf_counter() - t0, 2),
        }


def load_diseases() -> list[dict]:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    return [{"no": d["disease_no"], "title": d["title"]} for d in manifest["diseases"]]
