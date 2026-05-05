# Day 19 GraphRAG Knowledge Graph

**Học viên: Đặng Văn Minh**
**MSHV: 2A202600027**

Đây là bài nộp Lab Day 19: xây dựng một hệ thống GraphRAG end-to-end trên một corpus nhỏ về các công ty công nghệ.

Dự án minh họa các nội dung chính:

- Trích xuất bộ ba tri thức từ văn bản Markdown bằng LLM hoặc bằng chế độ deterministic/offline.
- Chuẩn hóa entity/relation và khử trùng lặp triples.
- Xây dựng knowledge graph bằng NetworkX.
- Trực quan hóa graph bằng Matplotlib.
- Xây dựng baseline Flat RAG bằng TF-IDF retrieval.
- Xây dựng GraphRAG với entity matching và duyệt graph 2-hop.
- So sánh Flat RAG và GraphRAG trên bộ benchmark 20 câu hỏi.
- Ghi nhận runtime, token usage và chi phí ước tính khi indexing.

## Lệnh Chấm Nhanh

Từ thư mục gốc của repository, chạy:

```bash
python main.py
```

Lệnh này sinh lại toàn bộ artifact cần nộp bằng chế độ deterministic/offline. Chế độ này không cần OpenAI API key, phù hợp để giảng viên chấm lại nhanh và ổn định.

Nếu muốn chạy một lần thực nghiệm thật với LLM, tạo file `.env` và chạy:

```bash
cp .env.example .env
# sửa .env và điền OPENAI_API_KEY
python main.py --openai
```

`python main.py --openai` sẽ gọi OpenAI cho bước triple extraction và answer generation trong benchmark, nên có thể phát sinh chi phí API.

## Giảng Viên Nên Kiểm Tra Gì

Sau khi chạy `python main.py`, xem các file sau:

- `outputs/submission_status.md`: trạng thái tổng thể của pipeline và checklist artifact.
- `outputs/benchmark_report.md`: báo cáo benchmark dễ đọc, có phần GraphRAG cải thiện ở đâu so với Flat RAG.
- `outputs/benchmark_results.csv`: bảng benchmark thô.
- `outputs/knowledge_graph.png`: ảnh knowledge graph đã sinh.
- `outputs/cost_report.json`: báo cáo runtime/token/cost của bước extraction.

Kết quả kỳ vọng ở chế độ deterministic:

```text
Flat RAG accuracy: 15/20 = 75.0%
GraphRAG accuracy: 20/20 = 100.0%
```

Điểm cải thiện chính nằm ở nhóm câu hỏi adversarial. Flat RAG thường retrieve được đoạn văn liên quan nhưng không kiểm tra quan hệ rõ ràng, trong khi GraphRAG dùng triples và graph traversal để trả lời chính xác hơn.

## Cài Đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu chạy chế độ OpenAI thật:

```bash
cp .env.example .env
```

Sau đó sửa `.env`:

```bash
OPENAI_API_KEY=your_real_key_here
OPENAI_EXTRACTION_MODEL=gpt-4.1-mini
OPENAI_ANSWER_MODEL=gpt-4.1-mini
```

## Cấu Trúc Repository

```text
.
├── data/
│   ├── raw/
│   │   └── tech_company_corpus.md
│   ├── processed/
│   │   ├── chunks.json
│   │   ├── triples.json
│   │   └── normalized_triples.json
│   └── benchmark/
│       └── questions.json
├── docs/
│   ├── Lab_day_19.md
│   └── implementation_spec.md
├── outputs/
│   ├── benchmark_report.md
│   ├── benchmark_results.csv
│   ├── cost_report.json
│   ├── knowledge_graph.png
│   ├── submission_status.json
│   └── submission_status.md
├── src/
│   ├── config.py
│   ├── corpus.py
│   ├── evaluate.py
│   ├── extract_triples.py
│   ├── flat_rag.py
│   ├── graph_rag.py
│   ├── graph_store.py
│   └── visualize.py
├── main.py
├── .env.example
├── requirements.txt
└── README.md
```

Các file sinh ra trong `outputs/` được ignore bởi Git theo mặc định. Nếu cần nộp kèm artifact đã sinh, có thể dùng `git add -f`.

## Các Phase Trong Pipeline

`main.py` chạy các phase sau và in rõ `input`, `process`, `expected output`, `status`, `duration` trên terminal:

1. Chunking corpus và triple extraction.
2. Chuẩn hóa triples và xây dựng NetworkX graph.
3. Trực quan hóa knowledge graph.
4. Chạy benchmark so sánh Flat RAG và GraphRAG.
5. Sinh submission status report.

## Các Lệnh Chạy Riêng Lẻ

Chạy extraction deterministic:

```bash
python -m src.extract_triples --offline
```

Chuẩn hóa triples và in graph summary:

```bash
python -m src.graph_store
```

Sinh ảnh graph:

```bash
python -m src.visualize
```

Hỏi Flat RAG:

```bash
python -m src.flat_rag "Which company acquired Instagram?"
```

Hỏi GraphRAG:

```bash
python -m src.graph_rag "Which AI company is connected to Tesla through Elon Musk?"
```

Chạy riêng benchmark:

```bash
python -m src.evaluate
```

## Ghi Chú

- Chế độ deterministic giúp giảng viên chấm lại bài mà không cần API credential.
- Chế độ OpenAI dùng để chạy thực nghiệm thật với LLM theo yêu cầu bài lab.
- API key được đọc từ `.env` và không được commit.
- Neo4j export là phần mở rộng optional trong spec, không bắt buộc cho main path dùng NetworkX.
