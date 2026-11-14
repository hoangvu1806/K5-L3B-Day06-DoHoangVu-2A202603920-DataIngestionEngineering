# RHM RAG Chatbot

Chatbot tra cứu [tài liệu hướng dẫn Răng Hàm Mặt](../data/manifest.json) dựa trên pipeline Day 06.

## Stack
- **UI:** FastAPI + Jinja2 + HTML tĩnh (cùng kiểu template như Day 04 ReAct Loop Agent)
- **Retrieval:** dùng lại `starter/src/{embeddings,store}.py` — Qdrant + `google/gemini-embedding-2` (3072 chiều), chiến lược C (B + filter `disease_no` khi người dùng chọn bệnh)
- **Sinh câu trả lời:** `deepseek/deepseek-v4.1-flash` qua OpenRouter (giống Day 04)
- **Palette:** teal + white, thêm biểu tượng caduceus, disclaimer y tế

## Chạy
Từ thư mục gốc repo:
```
python -m pip install -r starter/requirements.txt -r chatbot/requirements.txt
python -m uvicorn chatbot.app:app --reload --port 8000
```

Mở http://localhost:8000. Cần `starter/.env` đã có `QDRANT_URL`, `QDRANT_API_KEY`, `OPENROUTER_API_KEY`, và các collection Qdrant `day06_rhm_a`/`day06_rhm_b` đã nạp (bằng `python -m src.index` từ `starter/`).

Tùy chọn trong `.env`:
- `CHAT_MODEL=deepseek/deepseek-v4.1-flash` (mặc định; có thể đổi sang model OpenRouter khác hỗ trợ chat completion)

## API
`POST /api/ask` nhận JSON:
```json
{"query": "Khi nào nhổ răng khôn?", "disease_no": 1, "top_k": 5}
```
Trả về: `answer`, `sources` (top-k chunk với citation, trang, bệnh, section), `strategy`, chi phí, thời gian.

## Nguyên tắc trả lời (system prompt)
1. Chỉ dùng thông tin trong các đoạn tài liệu được cung cấp.
2. Trích nguồn [#n] theo số thứ tự chunk.
3. Thiếu thông tin thì trả lời "Không rõ, tài liệu không nêu".
4. Không chẩn đoán cá nhân — đây là công cụ tra cứu tài liệu.

## Kết quả thử (2026-11-14)
| Câu hỏi | Chiến lược | Kết quả |
|---|---|---|
| Khi nào nhổ răng khôn mọc lệch? | B (không lọc) | Trả lời "nhổ càng sớm càng tốt…" [#2], 3 nguồn đều bệnh 1 (tr.8–9); 4,1 s; $0.0003 |
| Chỉ định rạch dẫn lưu đường trong miệng? (lọc bệnh 30) | C | "khi áp xe tiến triển về phía niêm mạc miệng" [#2], 3 nguồn đều bệnh 30 |
| Quyết định có hiệu lực từ khi nào? | B | "kể từ ngày ký ban hành… 28/7/2015" [#1] (trang Quyết định, text từ OCR) |
| Thuốc nào trị đau bụng kinh? (ngoài tài liệu) | B | Từ chối đúng: "Không rõ, tài liệu không nêu." |

Lưu ý khi thử bằng `curl` trên Git Bash (Windows): một số ký tự tiếng Việt có thể bị gửi sai mã hóa và server trả 400 "error parsing the body". Gửi bằng trình duyệt hoặc Python `requests` thì không bị.
