# Implementation Spec: Day 19 GraphRAG Knowledge Graph

## 1. Project Goal

Build a simple but complete end-to-end GraphRAG system for a small technology company corpus.

The implementation will:

- Use NetworkX as the main graph engine.
- Use Matplotlib to visualize the knowledge graph.
- Optionally export graph data to Neo4j if time allows.
- Compare a simple Flat RAG baseline against GraphRAG.
- Use the real OpenAI API for extraction and/or answer generation.
- Track token usage and runtime cost during graph construction.

## 2. Chosen Scope

### Main Path

The main implementation will use:

- Python scripts, not notebooks.
- A small self-created corpus about technology companies.
- A hybrid extraction approach:
  - Raw corpus is written manually or crawled and saved as Markdown or text.
  - LLM extraction is used to convert text chunks into structured triples.
  - Extracted triples are reviewed, normalized, and saved for stable reruns.
- NetworkX for graph construction and multi-hop traversal.
- A simple Flat RAG baseline using local retrieval.
- OpenAI API for LLM-based extraction and final answer generation.

### Optional Path

If the main pipeline is complete and there is enough time:

- Export entities and relations to Neo4j.
- Add a short Cypher-based graph loading script.
- Capture a Neo4j Browser or Bloom screenshot for the report.

## 3. Proposed Repository Structure

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
│   ├── knowledge_graph.png
│   ├── benchmark_results.csv
│   └── cost_report.json
├── src/
│   ├── config.py
│   ├── corpus.py
│   ├── extract_triples.py
│   ├── graph_store.py
│   ├── flat_rag.py
│   ├── graph_rag.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── main.py
├── .env.example
├── requirements.txt
└── README.md
```

The structure can be reduced if needed, but the implementation should keep data, source code, and outputs separated.

## 4. Data Plan

### Corpus

Create a small technology company corpus in `data/raw/tech_company_corpus.md`.

The corpus should include facts about:

- Companies
- Founders
- Products
- Acquisitions
- Investors
- Founded years
- Headquarters
- Parent companies
- Key partnerships

Target size:

- 20 to 40 short paragraphs.
- Around 10 to 15 companies.
- Enough overlap between entities to support multi-hop questions.

Example entity domains:

- OpenAI
- Microsoft
- Google
- DeepMind
- Meta
- Instagram
- WhatsApp
- Apple
- Anthropic
- Amazon
- NVIDIA
- Tesla

### Source Format

Use Markdown as the default corpus format because it is easy to read and edit.

Each paragraph should contain compact factual statements, for example:

```markdown
OpenAI was founded in 2015 by Sam Altman, Greg Brockman, Ilya Sutskever, John Schulman, Wojciech Zaremba, and Elon Musk. Microsoft invested in OpenAI and provides cloud infrastructure through Azure.
```

### Corpus Collection Strategy

Use a crawl-then-curate approach.

The corpus will be created through this flow:

```text
Wikipedia crawl summaries
        ↓
Manual cleanup and fact selection
        ↓
Curated Markdown corpus
        ↓
LLM triple extraction
        ↓
Normalization and deduplication
        ↓
NetworkX knowledge graph
```

The crawl step should collect short summaries from stable public pages such as Wikipedia. The cleanup step is required because raw crawled text may include noisy details that are not useful for relationship extraction.

Recommended crawl source:

- Wikipedia summaries for the selected technology companies.

Recommended crawl package:

- `wikipedia-api`

The crawler should:

- Use a clear `User-Agent`.
- Fetch only selected pages, not broad web crawls.
- Save raw crawled text for traceability.
- Avoid excessive requests.
- Keep only factual content useful for graph relations.

Add one optional raw crawl file:

```text
data/raw/wiki_crawled_corpus.md
```

Then manually curate the final corpus into:

```text
data/raw/tech_company_corpus.md
```

The curated corpus should be preferred by the GraphRAG pipeline. This keeps the lab reproducible and makes benchmark evaluation easier.

Suggested crawler script:

```text
src/crawl_wikipedia.py
```

This script should be optional. The final pipeline should still work if `data/raw/tech_company_corpus.md` already exists.

## 5. Extraction Plan

### Hybrid Triple Extraction

The extraction pipeline will:

1. Load the raw Markdown corpus.
2. Split it into small chunks.
3. Send each chunk to the OpenAI API.
4. Ask the model to extract structured triples.
5. Save raw extracted triples to `data/processed/triples.json`.
6. Normalize entity names and relation names.
7. Save final triples to `data/processed/normalized_triples.json`.

### Triple Format

Use JSON objects with explicit fields:

```json
{
  "subject": "OpenAI",
  "relation": "FOUNDED_BY",
  "object": "Sam Altman",
  "source_chunk_id": "chunk_001",
  "confidence": 0.95
}
```

### Relation Naming

Use uppercase snake case for relation names.

Suggested relation types:

- `FOUNDED_BY`
- `FOUNDED_IN`
- `CREATED_PRODUCT`
- `ACQUIRED`
- `ACQUIRED_BY`
- `INVESTED_IN`
- `BACKED_BY`
- `PARTNERED_WITH`
- `HEADQUARTERED_IN`
- `PARENT_COMPANY_OF`
- `SUBSIDIARY_OF`
- `DEVELOPED_BY`
- `LED_BY`

### Deduplication

Deduplication should handle:

- Repeated triples.
- Entity aliases, such as `Google` and `Google LLC`.
- Directional duplicates, such as `Microsoft INVESTED_IN OpenAI` and `OpenAI BACKED_BY Microsoft`.

For the first version, use a simple normalization map in code.

## 6. GraphRAG Pipeline

### Graph Construction

Build a directed NetworkX graph from normalized triples.

Each node should store:

- `name`
- `type`, if available
- optional metadata

Each edge should store:

- `relation`
- `source_chunk_id`
- optional confidence

### Query Flow

For each user question:

1. Extract the main entity or entities from the question.
2. Match extracted entities to graph nodes.
3. Traverse neighbors within 2 hops.
4. Convert relevant triples into textual context.
5. Send the question and graph context to the OpenAI API.
6. Return the final answer with supporting triples.

### Entity Matching

Start simple:

- Exact case-insensitive match.
- Alias map.
- Fallback substring match.

Optional improvement:

- Use embedding similarity for entity linking.

## 7. Flat RAG Baseline

The baseline should be intentionally simple.

Recommended implementation:

- Load raw corpus chunks.
- Retrieve top-k chunks using TF-IDF cosine similarity.
- Send retrieved chunks plus question to OpenAI API.

This avoids extra database setup while still giving a fair baseline for comparison.

Optional alternatives:

- FAISS with OpenAI embeddings.
- ChromaDB.

## 8. Benchmark Design

Create `data/benchmark/questions.json` with 20 questions.

Each question should include:

```json
{
  "id": "q001",
  "question": "Which company invested in OpenAI and also provides cloud infrastructure through Azure?",
  "expected_answer": "Microsoft",
  "type": "multi-hop",
  "required_entities": ["OpenAI", "Microsoft", "Azure"]
}
```

Question categories:

- 5 one-hop factual questions.
- 5 two-hop relationship questions.
- 5 comparison questions.
- 5 adversarial or ambiguity-prone questions where Flat RAG may hallucinate.

The benchmark should be designed so GraphRAG has a clear advantage on questions requiring entity relationships.

## 9. Evaluation Plan

For each benchmark question, run:

1. Flat RAG answer.
2. GraphRAG answer.
3. Compare both against the expected answer.

Save results to `outputs/benchmark_results.csv`.

Suggested columns:

- `question_id`
- `question`
- `expected_answer`
- `flat_rag_answer`
- `graph_rag_answer`
- `flat_rag_correct`
- `graph_rag_correct`
- `flat_rag_context`
- `graph_rag_context_triples`
- `notes`

Correctness can be manually reviewed first. If time allows, add an LLM-as-judge step.

## 10. Cost and Time Tracking

Track cost and runtime during graph construction, especially during LLM extraction.

Record:

- Number of corpus chunks.
- Number of extracted triples.
- Number of normalized triples.
- Extraction start time.
- Extraction end time.
- Total extraction duration.
- Input tokens.
- Output tokens.
- Total tokens.
- Estimated API cost.

Save this to `outputs/cost_report.json`.

Example:

```json
{
  "num_chunks": 24,
  "num_raw_triples": 105,
  "num_normalized_triples": 92,
  "extraction_duration_seconds": 48.2,
  "input_tokens": 18200,
  "output_tokens": 6400,
  "total_tokens": 24600,
  "estimated_cost_usd": 0.12
}
```

The final report should include a short discussion of:

- Token cost for indexing.
- Time cost for graph construction.
- Whether GraphRAG accuracy gains justify the extra indexing cost.

## 11. OpenAI API Usage

Use environment variables for API keys.

Required `.env` variable:

```bash
OPENAI_API_KEY=your_api_key_here
```

The code should not hardcode secrets.

Recommended model usage:

- Extraction: a low-cost model with structured JSON output.
- Answer generation: a stronger model if needed.

The exact model can be set in `src/config.py` or `.env`.

## 12. Visualization

Use Matplotlib with NetworkX layout functions.

The visualization should:

- Draw company, person, product, and location nodes with different colors if type metadata is available.
- Draw directed edges.
- Label nodes.
- Optionally label edge relations.
- Save output to `outputs/knowledge_graph.png`.

If the graph becomes too dense, create a filtered visualization around key companies.

## 13. Neo4j Export Optional Extension

If time allows, add:

- `src/export_neo4j.py`
- A `.env` configuration for:
  - `NEO4J_URI`
  - `NEO4J_USERNAME`
  - `NEO4J_PASSWORD`

The script should:

- Create or merge entity nodes.
- Create or merge relationship edges.
- Preserve relation types where possible.

This is optional and should not block the main NetworkX implementation.

## 14. Implementation Order

Recommended order:

1. Create project folders and `requirements.txt`.
2. Implement optional Wikipedia crawler.
3. Crawl selected company summaries into `data/raw/wiki_crawled_corpus.md`.
4. Manually clean and curate the final corpus into `data/raw/tech_company_corpus.md`.
5. Implement corpus loading and chunking.
6. Implement LLM triple extraction with token/time tracking.
7. Implement triple normalization and deduplication.
8. Build NetworkX graph.
9. Implement graph visualization.
10. Implement GraphRAG querying.
11. Implement Flat RAG baseline.
12. Create 20 benchmark questions.
13. Implement evaluation runner.
14. Generate benchmark results and cost report.
15. Write final analysis for the lab report.
16. Optionally export to Neo4j.

## 15. Definition of Done

The lab is complete when the repository contains:

- A runnable Python pipeline.
- A small technology company corpus.
- Extracted and normalized triples.
- A NetworkX knowledge graph.
- A saved graph visualization.
- A Flat RAG baseline.
- A GraphRAG query implementation.
- 20 benchmark questions.
- Benchmark comparison results.
- Token and time cost report.
- A short analysis explaining where GraphRAG improves over Flat RAG.

## 16. Current Decisions

Confirmed decisions:

- Main graph engine: NetworkX.
- Visualization: Matplotlib.
- Optional extension: Neo4j export.
- Corpus: crawl Wikipedia summaries, then manually curate a small technology company corpus.
- Corpus storage: Markdown preferred.
- Raw crawled corpus: `data/raw/wiki_crawled_corpus.md`.
- Final curated corpus: `data/raw/tech_company_corpus.md`.
- Extraction: hybrid approach with LLM extraction and normalized saved triples.
- Benchmark: diverse question set designed to compare Flat RAG and GraphRAG.
- Flat RAG: simple local baseline.
- LLM: real OpenAI API.
- Deliverable format: Python scripts.
- Cost tracking: token usage and runtime during graph construction.
