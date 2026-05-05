"""Shared configuration for the Day 19 GraphRAG lab.

The module keeps paths and model settings in one place so the rest of the
pipeline can stay small and easy to read during the lab walkthrough.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps scripts usable before install.
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RAW_CORPUS_PATH = RAW_DIR / "tech_company_corpus.md"
BENCHMARK_QUESTIONS_PATH = BENCHMARK_DIR / "questions.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
TRIPLES_PATH = PROCESSED_DIR / "triples.json"
NORMALIZED_TRIPLES_PATH = PROCESSED_DIR / "normalized_triples.json"
COST_REPORT_PATH = OUTPUTS_DIR / "cost_report.json"
KNOWLEDGE_GRAPH_IMAGE_PATH = OUTPUTS_DIR / "knowledge_graph.png"
BENCHMARK_RESULTS_PATH = OUTPUTS_DIR / "benchmark_results.csv"
BENCHMARK_REPORT_PATH = OUTPUTS_DIR / "benchmark_report.md"
SUBMISSION_STATUS_PATH = OUTPUTS_DIR / "submission_status.md"
SUBMISSION_STATUS_JSON_PATH = OUTPUTS_DIR / "submission_status.json"


def load_environment() -> None:
    """Load local environment variables from `.env` when python-dotenv exists."""

    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")


load_environment()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EXTRACTION_MODEL = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-4.1-mini")
OPENAI_ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4.1-mini")

# Conservative default estimates used only for the lab cost report.
# You can update these values if your selected model has different pricing.
MODEL_PRICES_USD_PER_1M_TOKENS = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
