# Submission Status

- Overall status: OK
- Total runtime: 153.69s
- Flat RAG accuracy: 17/20 = 85.0%
- GraphRAG accuracy: 18/20 = 90.0%

## Pipeline Steps

| Step | Status | Duration |
|---|---|---:|
| 1. Corpus chunking and triple extraction | OK | 94.82s |
| 2. Triple normalization and NetworkX graph build | OK | 0.00s |
| 3. Knowledge graph visualization | OK | 1.24s |
| 4. Flat RAG vs GraphRAG benchmark | OK | 57.62s |

## Submission Artifacts

| Artifact | Exists | Size |
|---|---:|---:|
| `/Users/dangminh/Desktop/Vin-PracticalAI/assignments/day19-GraphRAG/Day19-GraphRAG-KnowledgeGraph/data/processed/chunks.json` | yes | 10.5 KB |
| `/Users/dangminh/Desktop/Vin-PracticalAI/assignments/day19-GraphRAG/Day19-GraphRAG-KnowledgeGraph/data/processed/triples.json` | yes | 18.9 KB |
| `/Users/dangminh/Desktop/Vin-PracticalAI/assignments/day19-GraphRAG/Day19-GraphRAG-KnowledgeGraph/data/processed/normalized_triples.json` | yes | 18.7 KB |
| `/Users/dangminh/Desktop/Vin-PracticalAI/assignments/day19-GraphRAG/Day19-GraphRAG-KnowledgeGraph/outputs/cost_report.json` | yes | 298 B |
| `/Users/dangminh/Desktop/Vin-PracticalAI/assignments/day19-GraphRAG/Day19-GraphRAG-KnowledgeGraph/outputs/knowledge_graph.png` | yes | 1.1 MB |
| `/Users/dangminh/Desktop/Vin-PracticalAI/assignments/day19-GraphRAG/Day19-GraphRAG-KnowledgeGraph/outputs/benchmark_results.csv` | yes | 50.1 KB |
| `/Users/dangminh/Desktop/Vin-PracticalAI/assignments/day19-GraphRAG/Day19-GraphRAG-KnowledgeGraph/outputs/benchmark_report.md` | yes | 4.4 KB |
| `/Users/dangminh/Desktop/Vin-PracticalAI/assignments/day19-GraphRAG/Day19-GraphRAG-KnowledgeGraph/outputs/submission_status.md` | yes | 2.1 KB |
| `/Users/dangminh/Desktop/Vin-PracticalAI/assignments/day19-GraphRAG/Day19-GraphRAG-KnowledgeGraph/outputs/submission_status.json` | yes | 4.9 KB |

## Notes For Grading

- `python main.py` regenerates all deterministic offline artifacts.
- `python main.py --openai` uses OpenAI for extraction and answer generation when `OPENAI_API_KEY` is configured.
- Benchmark CSV is the raw output; benchmark Markdown is the readable report.
