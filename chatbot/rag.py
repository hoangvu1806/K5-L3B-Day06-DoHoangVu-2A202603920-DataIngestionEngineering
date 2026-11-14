"""Thin RAG wrapper around the Day 06 pipeline.

Reuses the pipeline's embedding provider and Qdrant client (strategy B / C):
  - retrieve top-k chunks from collection `day06_rhm_b`, with optional
    `disease_no` filter supplied by the UI (this is the Day 06 'strategy C');
  - ask the chat model for a grounded answer, told to cite [#n] by chunk index
    and to say 'Không rõ' when the context is insufficient.

Latency: one keep-alive HTTP session per provider, and warmup() at server
start opens the embedding, Qdrant and chat connections before the first user.
answer_stream() yields the sources first, then answer tokens as they arrive.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import requests

# Reuse the Day 06 pipeline modules without copying code.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from src.embeddings import OpenRouterEmbedder  # noqa: E402
from src.store import QdrantStore              # noqa: E402

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
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
        # Hidden reasoning adds ~300-400 tokens before the first visible word; a
        # grounded extract-and-cite answer does not need it. CHAT_REASONING=on re-enables.
        self.reasoning = os.getenv("CHAT_REASONING", "off").lower() in ("1", "on", "true")
        self.chat = requests.Session()
        self.chat.headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"
        self.warm: dict[str, Any] = {"ready": False}

    # ------------------------------------------------------------------ warmup
    def warmup(self) -> dict[str, Any]:
        """Open every connection once so the first real question is fast."""
        t = {}
        t0 = time.perf_counter()
        vec = self.embedder.embed_query("khởi động hệ thống")
        t["embed_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        self.store.search("B", vec, 1)
        self.store.search("C", vec, 1, {"disease_no": 1})
        t["qdrant_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        resp = self.chat.post(CHAT_URL, json={"model": self.chat_model, "max_tokens": 1, **self._reasoning(),
                                              "messages": [{"role": "user", "content": "ping"}]},
                              timeout=60)
        resp.raise_for_status()
        t["chat_s"] = round(time.perf_counter() - t0, 2)
        self.warm = {"ready": True, **t}
        return self.warm

    def _reasoning(self) -> dict:
        return {} if self.reasoning else {"reasoning": {"enabled": False}}

    # --------------------------------------------------------------- retrieval
    def retrieve(self, question: str, disease_no: int | None, top_k: int) -> list[dict]:
        strategy = "C" if disease_no else "B"
        filters = {"disease_no": disease_no} if disease_no else None
        vec = self.embedder.embed_query(question)
        hits = self.store.search(strategy, vec, top_k, filters)
        for i, h in enumerate(hits, 1):
            h["citation"] = f"#{i}"
        return hits

    @staticmethod
    def _context(hits: list[dict]) -> str:
        parts = []
        for h in hits:
            md = h["metadata"]
            label = []
            if md.get("disease"):
                label.append(f"Bệnh {md.get('disease_no')}. {md['disease']}")
            elif md.get("special_id"):
                label.append(md.get("section") or md["special_id"])
            if md.get("section") and md.get("disease"):
                label.append(md["section"])
            if md.get("subsection"):
                label.append(md["subsection"])
            pages = md.get("pages") or [md.get("page")]
            parts.append(f"[{h['citation']}] ({' > '.join(label) or '—'}; trang {pages})\n{h['text']}")
        return "\n\n".join(parts)

    @staticmethod
    def _sources(hits: list[dict]) -> list[dict]:
        return [{
            "citation": h["citation"], "chunk_id": h["chunk_id"], "score": round(h["score"], 3),
            "disease_no": h["metadata"].get("disease_no"), "disease": h["metadata"].get("disease"),
            "section": h["metadata"].get("section"), "subsection": h["metadata"].get("subsection"),
            "special_id": h["metadata"].get("special_id"),
            "pages": h["metadata"].get("pages") or [h["metadata"].get("page")],
            "text": h["text"],
        } for h in hits]

    def _messages(self, question: str, hits: list[dict]) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Câu hỏi: {question}\n\nCác đoạn tài liệu tham khảo:\n\n{self._context(hits)}\n\n"
                "Dựa trên các đoạn trên, hãy trả lời câu hỏi, trích nguồn [#n]."},
        ]

    # -------------------------------------------------------------- streaming
    def answer_stream(self, question: str, disease_no: int | None = None,
                      top_k: int = DEFAULT_TOP_K) -> Iterator[tuple[str, dict]]:
        """Yield ('meta', ...), ('sources', ...), ('delta', ...)*, ('done', ...)."""
        t0 = time.perf_counter()
        strategy = "C" if disease_no else "B"
        yield "meta", {"strategy": strategy, "disease_no": disease_no, "top_k": top_k,
                       "model": self.chat_model, "embedding_model": self.embedder.model_name,
                       "reasoning": self.reasoning}
        hits = self.retrieve(question, disease_no, top_k)
        messages = self._messages(question, hits)
        yield "sources", {
            "sources": self._sources(hits),
            "retrieval_s": round(time.perf_counter() - t0, 2),
            # Exactly what was used: the user's question is embedded verbatim
            # (no LLM query rewriting), plus the prompt sent to the chat model.
            "retrieval": {
                "query": question,
                "query_rewritten": False,
                "embedding_model": self.embedder.model_name,
                "strategy": strategy,
                "collection": self.store.collection(strategy),
                "filter": {"disease_no": disease_no} if disease_no else None,
                "top_k": top_k,
                "search": "cosine, exact",
            },
            "prompt": messages,
        }

        payload = {"model": self.chat_model, "messages": messages,
                   "temperature": 0.2, "stream": True, "usage": {"include": True},
                   **self._reasoning()}
        first_token_s = None
        usage: dict[str, Any] = {}
        with self.chat.post(CHAT_URL, json=payload, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:300]}")
            resp.encoding = "utf-8"
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue          # blank keep-alives and ': OPENROUTER PROCESSING' comments
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(f"OpenRouter error: {chunk['error']}")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        if first_token_s is None:
                            first_token_s = round(time.perf_counter() - t0, 2)
                        yield "delta", {"text": text}
        yield "done", {"tokens": usage.get("total_tokens"), "cost_usd": usage.get("cost"),
                       "first_token_s": first_token_s,
                       "seconds": round(time.perf_counter() - t0, 2)}

    def answer(self, question: str, disease_no: int | None = None,
               top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """Non-streaming wrapper (same output shape as before)."""
        out: dict[str, Any] = {"answer": ""}
        for event, data in self.answer_stream(question, disease_no, top_k):
            if event == "delta":
                out["answer"] += data["text"]
            else:
                out.update(data)
        return out


def load_diseases() -> list[dict]:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    return [{"no": d["disease_no"], "title": d["title"]} for d in manifest["diseases"]]
