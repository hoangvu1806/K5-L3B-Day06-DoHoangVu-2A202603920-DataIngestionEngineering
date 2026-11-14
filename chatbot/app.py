"""FastAPI web UI for the Day 06 RAG chatbot.

Run from repo root:
    python -m uvicorn chatbot.app:app --port 8000

At startup the RAG engine is created and warmed up (embedding, Qdrant and chat
connections opened once), so the first question answers without cold start.
"""
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .benchmark import load_benchmark
from .rag import RagEngine, load_diseases

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("uvicorn.error")

_engine: RagEngine | None = None
_lock = threading.Lock()
_warm_error: str | None = None


def engine() -> RagEngine:
    global _engine
    with _lock:
        if _engine is None:
            _engine = RagEngine()
        return _engine


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _warm_error
    try:
        info = engine().warmup()
        log.info("RAG warmup done: embed %ss, qdrant %ss, chat %ss",
                 info["embed_s"], info["qdrant_s"], info["chat_s"])
    except Exception as exc:  # noqa: BLE001 - keep serving; report in /api/health
        _warm_error = f"{type(exc).__name__}: {exc}"
        log.warning("RAG warmup failed: %s", _warm_error)
    yield


app = FastAPI(title="RHM RAG — Tra cứu hướng dẫn Răng Hàm Mặt", lifespan=lifespan)
templates = Jinja2Templates(directory=str(ROOT / "templates"))


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    disease_no: int | None = Field(default=None, ge=1, le=57)
    top_k: int = Field(default=5, ge=1, le=10)


SAMPLES = [
    {"q": "Khi nào thì nhổ răng khôn mọc lệch?", "disease": 1},
    {"q": "Cân nặng tối thiểu để phẫu thuật khe hở môi là bao nhiêu?", "disease": 23},
    {"q": "Carbamazepine dùng cho bệnh gì và cách dùng thế nào?", "disease": 48},
    {"q": "Trong viêm quanh Implant, xương tiêu đến mức nào thì nên tháo Implant?", "disease": 52},
    {"q": "Quai bị do virus nào gây ra và biến chứng thường gặp?", "disease": 42},
    {"q": "Quyết định ban hành tài liệu này có hiệu lực từ khi nào?", "disease": None},
]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    try:
        diseases = load_diseases()
    except FileNotFoundError:
        diseases = []
    return templates.TemplateResponse(request, "index.html",
                                      {"samples": SAMPLES, "diseases": diseases})


@app.get("/api/benchmark")
def benchmark():
    """Frozen Day 06 benchmark: per-strategy scores and top-3 traces of all 24 questions."""
    try:
        return load_benchmark()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Thiếu file kết quả benchmark: {exc.filename}")


@app.get("/api/health")
def health():
    warm = _engine.warm if _engine else {"ready": False}
    return {**warm, "error": _warm_error}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/ask/stream")
def ask_stream(body: AskRequest):
    """Server-Sent Events: meta, sources, delta*, done (or error)."""
    def events():
        try:
            for event, data in engine().answer_stream(body.query.strip(), body.disease_no, body.top_k):
                yield _sse(event, data)
        except Exception as exc:  # noqa: BLE001 - surface the error to the UI
            yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/ask")
def ask(body: AskRequest):
    """Non-streaming variant (whole answer in one JSON)."""
    try:
        return engine().answer(body.query.strip(), body.disease_no, body.top_k)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
