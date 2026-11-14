# Công nghệ và thiết lập đã dùng

Tài liệu này ghi lại các lựa chọn thực tế của bài nộp (yêu cầu của đề: ghi phiên bản, tên model, số chiều, cách chạy, chi phí; giữ cùng model, store và metric giữa ba cấu hình). Hướng dẫn chạy chi tiết ở [`starter/README.md`](../starter/README.md); cấu hình pipeline và kết quả ở [`report/individual/2A202603920_DoHoangVu.md`](../report/individual/2A202603920_DoHoangVu.md).

## Tóm tắt

| Thành phần | Lựa chọn |
|---|---|
| Ngôn ngữ / môi trường | Python 3.11.9, môi trường mặc định (không dùng venv riêng) |
| Trích text PDF | PyMuPDF (`fitz`) 1.26.4 |
| OCR cho trang scan | EasyOCR 1.7.2 (chỉ chạy một lần cho trang 1 — bản scan Quyết định) |
| Embedding | `google/gemini-embedding-2` qua OpenRouter, **3072 chiều** |
| Query / document | Embed cùng một cách (OpenRouter không có tham số task type) |
| Vector store | **Qdrant tự host 1.19.1** (https://qdrant.vectorium.space) |
| Client | `qdrant-client` 1.15.1, qua HTTPS cổng 443 |
| Metric / search | Cosine, **exact search** (`SearchParams(exact=True)`) |
| Payload index | `disease_no` (integer), `document_id`, `special_id`, `section` (keyword) |
| Metadata filter | `disease_no` (1–57, lấy từ mục lục PDF) |
| Corpus | PDF 208 trang, 194 trang nội dung (57 bệnh) |

## Embedding model

**Lý do chọn OpenRouter + `google/gemini-embedding-2`:**
- Tiếng Việt chất lượng cao, 3072 chiều.
- OpenRouter báo chi phí theo từng call (trường `usage.cost`), tiện ghi tổng chi phí vào báo cáo.
- Chi phí thực tế: nạp 1073 chunk hết $0.03, mỗi câu query $0.000004. Tổng chi phí phát triển $0.06.

**Đã cân nhắc:**
- API Gemini trực tiếp: key có nhưng hết credit (HTTP 402).
- Sentence Transformers local: khả thi, nhưng chọn API để kết quả ổn định hơn trên tiếng Việt chuyên ngành.

**Chống fallback thầm lặng:** `starter/src/embeddings.py` chỉ retry với lỗi tạm thời (429, 5xx, timeout); lỗi khác raise `EmbeddingError`. Không bao giờ dùng vector giả.

## Vector store

**Lý do chọn Qdrant tự host:**
- Người dùng đã có server tự host sẵn, không cần dựng local.
- Payload index và filter tương tự WHERE của SQL, đủ cho metadata filter của cấu hình C.
- `exact=True` cho kết quả tất định, filter không bị thiếu ứng viên như khi dùng HNSW xấp xỉ.

**Đã cân nhắc:**
- pgvector (starter gốc): cần Docker, phức tạp hơn và không được dùng sẵn hạ tầng.
- Chroma: client local lưu đĩa, nhưng không cần vì đã có Qdrant.

**Lưu ý triển khai:**
- `qdrant-client` mặc định dùng cổng 6333; server sau HTTPS proxy cần `port=443`. `starter/src/store.py` tự suy ra từ URL.
- Có cảnh báo lệch phiên bản (client 1.15.1 vs server 1.19.1), nhưng mọi lệnh cơ bản (create, upsert, search, filter) vẫn chạy.
- Một collection cho cấu hình A (`day06_rhm_a`, 547 điểm), một cho B (`day06_rhm_b`, 526 điểm). Cấu hình C đọc chung collection B.

## OCR

Trang 1 của PDF là **bản scan Quyết định 3108/QĐ-BYT** (không có lớp text). Đã xử lý:
- EasyOCR tiếng Việt, GPU 4 GB cần `canvas_size=1700, mag_ratio=1.0` để không hết RAM.
- Kết quả thô lưu trong [`data/ocr/page_001.json`](../data/ocr/page_001.json) (69 box, confidence trung bình 0.81).
- 26 sửa lỗi OCR đã đối chiếu ảnh scan (số 3108, ngày 28/7 viết tay, dấu câu, thứ tự cột) trong [`data/ocr/page_001_corrections.json`](../data/ocr/page_001_corrections.json). Code bắt buộc mỗi sửa lỗi phải khớp đúng 1 lần, không khớp thì raise.
- OCR và sửa lỗi là bước một lần; `starter/src/ingestion.py` đọc kết quả đã commit, không cần cài EasyOCR để chạy lại pipeline.

## Giữ đồng nhất giữa 3 cấu hình

Theo yêu cầu đề:
- **Cùng corpus:** `data/manifest.json` + `data/ocr/`.
- **Cùng model embedding, cùng cách embed query/document:** OpenRouter gọi `/embeddings` với cùng model cho cả hai.
- **Cùng vector store, cùng metric:** Qdrant, cosine, exact search.
- **Cùng bộ câu hỏi:** 24 câu đóng băng trong `benchmark/questions.json`.
- **Thay đổi có kiểm soát:** A↔B đổi cách chunk (cùng model/store); B↔C đổi filter (cùng chunk/model/store).
