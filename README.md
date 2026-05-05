# Day19 GraphRAG Knowledge Graph

Lab Day 19 for building a small end-to-end GraphRAG system over a curated technology company corpus.

## Goal

This project follows `docs/implementation_spec.md` and focuses on a simple, reproducible Python pipeline:

- Curate a small Markdown corpus about technology companies.
- Extract subject-relation-object triples.
- Normalize triples and build a NetworkX knowledge graph.
- Visualize the graph with Matplotlib.
- Compare a simple Flat RAG baseline with GraphRAG on benchmark questions.
- Track indexing cost and runtime.

## Project Structure

```text
.
├── data/
│   ├── raw/          # Curated and optional crawled corpus files
│   ├── processed/    # Chunks, extracted triples, normalized triples
│   └── benchmark/    # Evaluation questions
├── docs/             # Lab guide and implementation spec
├── outputs/          # Generated graph images, benchmark CSV, cost report
├── src/              # Python implementation
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your real `OPENAI_API_KEY` to `.env` if you want to run LLM extraction and answer generation.

## Planned Commands

The implementation will be added phase by phase. The final workflow will look like:

```bash
python -m src.main build
python -m src.main visualize
python -m src.main evaluate
```

Generated files under `outputs/` are ignored by Git except for `.gitkeep`.

## Current Extraction Command

Phase 3 includes corpus chunking and raw triple extraction:

```bash
python -m src.extract_triples --offline
```

Use `--offline` for deterministic lab data. Without `--offline`, the script will call OpenAI when `OPENAI_API_KEY` is set.

Generated artifacts:

- `data/processed/chunks.json`
- `data/processed/triples.json`
- `outputs/cost_report.json`

Phase 4 normalizes triples and builds a NetworkX graph summary:

```bash
python -m src.graph_store
```

Generated artifact:

- `data/processed/normalized_triples.json`

Phase 5 visualizes the knowledge graph:

```bash
python -m src.visualize
```

Use `--full` to render every node and edge. The default renders a focused graph that is easier to inspect in the lab report.

Generated artifact:

- `outputs/knowledge_graph.png`

Phase 6 runs a simple Flat RAG baseline with TF-IDF retrieval:

```bash
python -m src.flat_rag "Which company acquired Instagram?"
```

Add `--openai` to use OpenAI for answer generation when `OPENAI_API_KEY` is configured.
