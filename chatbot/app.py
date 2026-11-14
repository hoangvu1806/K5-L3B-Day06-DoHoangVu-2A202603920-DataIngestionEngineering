"""FastAPI web UI for the Day 06 RAG chatbot.

Run from repo root:
    python -m uvicorn chatbot.app:app --reload --port 8000
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .rag import RagEngine, load_diseases

ROOT = Path(__file__).resolve().parent

app = FastAPI(title="RHM RAG — Tra cứu hướng dẫn Răng Hàm Mặt")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
_engine: RagEngine | None = None


def engine() -> RagEngine:
    global _engine
    if _engine is None:
        _engine = RagEngine()
    return _engine


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


@app.post("/api/ask")
def ask(body: AskRequest):
    try:
        result = engine().answer(body.query.strip(), body.disease_no, body.top_k)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result
