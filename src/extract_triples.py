"""Triple extraction pipeline for the GraphRAG lab.

The preferred path uses the OpenAI API to extract triples from each corpus
chunk. For reproducible lab demos, the script also includes a curated offline
fallback that mirrors the facts in `data/raw/tech_company_corpus.md`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import (
    COST_REPORT_PATH,
    MODEL_PRICES_USD_PER_1M_TOKENS,
    OPENAI_API_KEY,
    OPENAI_EXTRACTION_MODEL,
    OUTPUTS_DIR,
    TRIPLES_PATH,
)
from src.corpus import CorpusChunk, chunk_corpus, save_chunks


ALLOWED_RELATIONS = {
    "FOUNDED_BY",
    "FOUNDED_IN",
    "CREATED_PRODUCT",
    "ACQUIRED",
    "ACQUIRED_BY",
    "INVESTED_IN",
    "BACKED_BY",
    "PARTNERED_WITH",
    "HEADQUARTERED_IN",
    "PARENT_COMPANY_OF",
    "SUBSIDIARY_OF",
    "DEVELOPED_BY",
    "LED_BY",
    "PROVIDES_INFRASTRUCTURE_FOR",
    "COMPETES_WITH",
}


@dataclass(frozen=True)
class Triple:
    """A raw extracted knowledge graph triple."""

    subject: str
    relation: str
    object: str
    source_chunk_id: str
    confidence: float = 0.9


@dataclass
class ExtractionUsage:
    """Token and timing metadata for the extraction cost report."""

    input_tokens: int = 0
    output_tokens: int = 0
    extraction_mode: str = "offline_curated"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate_cost_usd(model: str, usage: ExtractionUsage) -> float:
    """Estimate API cost from token usage and configured model pricing."""

    prices = MODEL_PRICES_USD_PER_1M_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    return round(
        (usage.input_tokens / 1_000_000) * prices["input"]
        + (usage.output_tokens / 1_000_000) * prices["output"],
        6,
    )


def curated_triples_for_chunk(chunk: CorpusChunk) -> list[Triple]:
    """Return deterministic triples for the curated corpus.

    This fallback intentionally stays explicit. It makes the lab easy to review
    and gives the later normalization and GraphRAG phases stable input.
    """

    c = chunk.chunk_id
    triples_by_chunk: dict[str, list[tuple[str, str, str, float]]] = {
        "chunk_001": [
            ("OpenAI", "HEADQUARTERED_IN", "San Francisco", 0.98),
            ("OpenAI", "FOUNDED_IN", "2015", 0.98),
            ("OpenAI", "FOUNDED_BY", "Sam Altman", 0.98),
            ("OpenAI", "FOUNDED_BY", "Greg Brockman", 0.98),
            ("OpenAI", "FOUNDED_BY", "Ilya Sutskever", 0.98),
            ("OpenAI", "FOUNDED_BY", "John Schulman", 0.98),
            ("OpenAI", "FOUNDED_BY", "Wojciech Zaremba", 0.98),
            ("OpenAI", "FOUNDED_BY", "Elon Musk", 0.98),
        ],
        "chunk_002": [
            ("Microsoft", "INVESTED_IN", "OpenAI", 0.98),
            ("Microsoft", "PROVIDES_INFRASTRUCTURE_FOR", "OpenAI", 0.96),
            ("Microsoft", "CREATED_PRODUCT", "Azure", 0.95),
        ],
        "chunk_003": [
            ("OpenAI", "CREATED_PRODUCT", "ChatGPT", 0.98),
            ("OpenAI", "CREATED_PRODUCT", "GPT-4", 0.95),
        ],
        "chunk_004": [
            ("OpenAI", "LED_BY", "Sam Altman", 0.95),
            ("OpenAI", "FOUNDED_BY", "Greg Brockman", 0.95),
            ("OpenAI", "LED_BY", "Greg Brockman", 0.86),
        ],
        "chunk_005": [
            ("Microsoft", "HEADQUARTERED_IN", "Redmond, Washington", 0.98),
            ("Microsoft", "FOUNDED_IN", "1975", 0.98),
            ("Microsoft", "FOUNDED_BY", "Bill Gates", 0.98),
            ("Microsoft", "FOUNDED_BY", "Paul Allen", 0.98),
        ],
        "chunk_006": [
            ("Microsoft", "CREATED_PRODUCT", "Azure", 0.98),
            ("Microsoft", "CREATED_PRODUCT", "Windows", 0.98),
            ("Microsoft", "CREATED_PRODUCT", "Microsoft Office", 0.98),
            ("Microsoft", "ACQUIRED", "LinkedIn", 0.98),
            ("Microsoft", "ACQUIRED", "GitHub", 0.98),
        ],
        "chunk_007": [
            ("GitHub", "HEADQUARTERED_IN", "San Francisco", 0.95),
            ("GitHub", "SUBSIDIARY_OF", "Microsoft", 0.98),
            ("Microsoft", "PARENT_COMPANY_OF", "GitHub", 0.98),
        ],
        "chunk_008": [
            ("LinkedIn", "HEADQUARTERED_IN", "Sunnyvale, California", 0.95),
            ("LinkedIn", "SUBSIDIARY_OF", "Microsoft", 0.98),
            ("Microsoft", "PARENT_COMPANY_OF", "LinkedIn", 0.98),
        ],
        "chunk_009": [
            ("Google", "HEADQUARTERED_IN", "Mountain View, California", 0.98),
            ("Google", "FOUNDED_IN", "1998", 0.98),
            ("Google", "FOUNDED_BY", "Larry Page", 0.98),
            ("Google", "FOUNDED_BY", "Sergey Brin", 0.98),
        ],
        "chunk_010": [
            ("Alphabet", "PARENT_COMPANY_OF", "Google", 0.98),
            ("Google", "SUBSIDIARY_OF", "Alphabet", 0.98),
            ("Google", "CREATED_PRODUCT", "Google Search", 0.98),
            ("Google", "CREATED_PRODUCT", "Gmail", 0.98),
            ("Google", "CREATED_PRODUCT", "Android", 0.98),
            ("Google", "CREATED_PRODUCT", "Google Cloud", 0.98),
        ],
        "chunk_011": [
            ("Google", "ACQUIRED", "DeepMind", 0.98),
            ("DeepMind", "ACQUIRED_BY", "Google", 0.95),
            ("DeepMind", "HEADQUARTERED_IN", "London", 0.95),
        ],
        "chunk_012": [
            ("DeepMind", "CREATED_PRODUCT", "AlphaGo", 0.98),
            ("DeepMind", "CREATED_PRODUCT", "AlphaFold", 0.98),
        ],
        "chunk_013": [
            ("Google", "CREATED_PRODUCT", "Google Cloud", 0.95),
            ("Google Cloud", "COMPETES_WITH", "Microsoft Azure", 0.9),
            ("Google Cloud", "COMPETES_WITH", "Amazon Web Services", 0.9),
        ],
        "chunk_014": [
            ("Meta", "HEADQUARTERED_IN", "Menlo Park, California", 0.98),
            ("Meta", "FOUNDED_IN", "2004", 0.95),
            ("Meta", "FOUNDED_BY", "Mark Zuckerberg", 0.98),
            ("Meta", "FOUNDED_BY", "Eduardo Saverin", 0.98),
            ("Meta", "FOUNDED_BY", "Andrew McCollum", 0.98),
            ("Meta", "FOUNDED_BY", "Dustin Moskovitz", 0.98),
            ("Meta", "FOUNDED_BY", "Chris Hughes", 0.98),
        ],
        "chunk_015": [
            ("Meta", "ACQUIRED", "Instagram", 0.98),
            ("Instagram", "ACQUIRED_BY", "Meta", 0.95),
        ],
        "chunk_016": [
            ("Meta", "ACQUIRED", "WhatsApp", 0.98),
            ("WhatsApp", "ACQUIRED_BY", "Meta", 0.95),
        ],
        "chunk_017": [
            ("Meta", "LED_BY", "Mark Zuckerberg", 0.98),
            ("Meta", "CREATED_PRODUCT", "Facebook", 0.95),
        ],
        "chunk_018": [
            ("Apple", "HEADQUARTERED_IN", "Cupertino, California", 0.98),
            ("Apple", "FOUNDED_IN", "1976", 0.98),
            ("Apple", "FOUNDED_BY", "Steve Jobs", 0.98),
            ("Apple", "FOUNDED_BY", "Steve Wozniak", 0.98),
            ("Apple", "FOUNDED_BY", "Ronald Wayne", 0.98),
        ],
        "chunk_019": [
            ("Apple", "CREATED_PRODUCT", "iPhone", 0.98),
            ("Apple", "CREATED_PRODUCT", "iPad", 0.98),
            ("Apple", "CREATED_PRODUCT", "Mac", 0.98),
            ("Apple", "CREATED_PRODUCT", "Apple Vision Pro", 0.98),
            ("Apple", "ACQUIRED", "Beats Electronics", 0.98),
        ],
        "chunk_020": [
            ("Anthropic", "HEADQUARTERED_IN", "San Francisco", 0.98),
            ("Anthropic", "FOUNDED_IN", "2021", 0.98),
            ("Anthropic", "FOUNDED_BY", "Dario Amodei", 0.98),
            ("Anthropic", "FOUNDED_BY", "Daniela Amodei", 0.98),
        ],
        "chunk_021": [
            ("Anthropic", "CREATED_PRODUCT", "Claude", 0.98),
            ("Amazon", "INVESTED_IN", "Anthropic", 0.98),
            ("Amazon", "PROVIDES_INFRASTRUCTURE_FOR", "Anthropic", 0.96),
            ("Amazon", "CREATED_PRODUCT", "Amazon Web Services", 0.95),
        ],
        "chunk_022": [
            ("Amazon", "HEADQUARTERED_IN", "Seattle, Washington", 0.98),
            ("Amazon", "FOUNDED_IN", "1994", 0.98),
            ("Amazon", "FOUNDED_BY", "Jeff Bezos", 0.98),
        ],
        "chunk_023": [
            ("Amazon", "CREATED_PRODUCT", "Amazon Web Services", 0.98),
            ("Amazon", "ACQUIRED", "Twitch", 0.98),
            ("Amazon", "ACQUIRED", "Whole Foods Market", 0.98),
        ],
        "chunk_024": [
            ("NVIDIA", "HEADQUARTERED_IN", "Santa Clara, California", 0.98),
            ("NVIDIA", "FOUNDED_IN", "1993", 0.98),
            ("NVIDIA", "FOUNDED_BY", "Jensen Huang", 0.98),
            ("NVIDIA", "FOUNDED_BY", "Chris Malachowsky", 0.98),
            ("NVIDIA", "FOUNDED_BY", "Curtis Priem", 0.98),
        ],
        "chunk_025": [
            ("NVIDIA", "CREATED_PRODUCT", "CUDA", 0.98),
            ("NVIDIA", "PARTNERED_WITH", "Microsoft", 0.92),
            ("NVIDIA", "PARTNERED_WITH", "Google", 0.92),
            ("NVIDIA", "PARTNERED_WITH", "Amazon", 0.92),
        ],
        "chunk_026": [
            ("Tesla", "HEADQUARTERED_IN", "Austin, Texas", 0.98),
            ("Tesla", "FOUNDED_IN", "2003", 0.98),
            ("Tesla", "FOUNDED_BY", "Martin Eberhard", 0.98),
            ("Tesla", "FOUNDED_BY", "Marc Tarpenning", 0.98),
        ],
        "chunk_027": [
            ("Elon Musk", "INVESTED_IN", "Tesla", 0.95),
            ("Tesla", "LED_BY", "Elon Musk", 0.98),
            ("Tesla", "CREATED_PRODUCT", "Model S", 0.98),
            ("Tesla", "CREATED_PRODUCT", "Model 3", 0.98),
            ("Tesla", "CREATED_PRODUCT", "Model X", 0.98),
            ("Tesla", "CREATED_PRODUCT", "Model Y", 0.98),
        ],
        "chunk_028": [
            ("OpenAI", "FOUNDED_BY", "Elon Musk", 0.98),
            ("Tesla", "LED_BY", "Elon Musk", 0.95),
        ],
        "chunk_029": [
            ("Microsoft", "PARTNERED_WITH", "NVIDIA", 0.96),
            ("Amazon", "PARTNERED_WITH", "NVIDIA", 0.96),
            ("Microsoft Azure", "PROVIDES_INFRASTRUCTURE_FOR", "AI workloads", 0.86),
            ("Amazon Web Services", "PROVIDES_INFRASTRUCTURE_FOR", "AI workloads", 0.86),
        ],
        "chunk_030": [
            ("Google", "PARTNERED_WITH", "NVIDIA", 0.96),
            ("NVIDIA", "PARTNERED_WITH", "Google Cloud", 0.92),
            ("NVIDIA", "PARTNERED_WITH", "Microsoft Azure", 0.92),
            ("NVIDIA", "PARTNERED_WITH", "Amazon Web Services", 0.92),
        ],
    }

    return [
        Triple(subject=subject, relation=relation, object=object_, source_chunk_id=c, confidence=confidence)
        for subject, relation, object_, confidence in triples_by_chunk.get(c, [])
    ]


def build_extraction_prompt(chunk: CorpusChunk) -> str:
    """Build the prompt used for LLM-based triple extraction."""

    relations = ", ".join(sorted(ALLOWED_RELATIONS))
    return (
        "Extract factual knowledge graph triples from the text below.\n"
        "Use only facts explicitly supported by the text.\n"
        f"Allowed relations: {relations}.\n"
        "Return JSON with a top-level key `triples`.\n"
        "Each triple must have subject, relation, object, and confidence.\n\n"
        f"Chunk ID: {chunk.chunk_id}\n"
        f"Text: {chunk.text}"
    )


def extract_with_openai(chunks: list[CorpusChunk]) -> tuple[list[Triple], ExtractionUsage]:
    """Extract triples with OpenAI structured JSON output."""

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError("The openai package is not installed. Run `pip install -r requirements.txt`.") from exc

    client = OpenAI(api_key=OPENAI_API_KEY)
    usage = ExtractionUsage(extraction_mode="openai")
    triples: list[Triple] = []

    for chunk in chunks:
        response = client.chat.completions.create(
            model=OPENAI_EXTRACTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You extract concise, accurate knowledge graph triples as JSON.",
                },
                {"role": "user", "content": build_extraction_prompt(chunk)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        message = response.choices[0].message.content or "{}"
        payload = json.loads(message)
        for item in payload.get("triples", []):
            relation = str(item.get("relation", "")).upper().strip()
            if relation not in ALLOWED_RELATIONS:
                continue
            triples.append(
                Triple(
                    subject=str(item.get("subject", "")).strip(),
                    relation=relation,
                    object=str(item.get("object", "")).strip(),
                    source_chunk_id=chunk.chunk_id,
                    confidence=float(item.get("confidence", 0.8)),
                )
            )

        if response.usage is not None:
            usage.input_tokens += response.usage.prompt_tokens
            usage.output_tokens += response.usage.completion_tokens

    return triples, usage


def extract_offline(chunks: list[CorpusChunk]) -> tuple[list[Triple], ExtractionUsage]:
    """Extract triples from the curated fallback map."""

    triples = [triple for chunk in chunks for triple in curated_triples_for_chunk(chunk)]
    return triples, ExtractionUsage(extraction_mode="offline_curated")


def save_triples(triples: list[Triple], path: Path = TRIPLES_PATH) -> None:
    """Persist raw triples as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(triple) for triple in triples], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_cost_report(
    *,
    chunks: list[CorpusChunk],
    triples: list[Triple],
    usage: ExtractionUsage,
    duration_seconds: float,
    path: Path = COST_REPORT_PATH,
) -> None:
    """Write the extraction cost and runtime report."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "num_chunks": len(chunks),
        "num_raw_triples": len(triples),
        "num_normalized_triples": None,
        "extraction_duration_seconds": round(duration_seconds, 3),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost_usd": estimate_cost_usd(OPENAI_EXTRACTION_MODEL, usage),
        "extraction_model": OPENAI_EXTRACTION_MODEL if usage.extraction_mode == "openai" else None,
        "extraction_mode": usage.extraction_mode,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def run_extraction(force_offline: bool = False) -> None:
    """Run corpus chunking and raw triple extraction end to end."""

    start_time = time.perf_counter()
    chunks = chunk_corpus()
    save_chunks(chunks)

    use_openai = bool(OPENAI_API_KEY) and not force_offline
    if use_openai:
        triples, usage = extract_with_openai(chunks)
    else:
        triples, usage = extract_offline(chunks)

    duration_seconds = time.perf_counter() - start_time
    save_triples(triples)
    save_cost_report(
        chunks=chunks,
        triples=triples,
        usage=usage,
        duration_seconds=duration_seconds,
    )

    print(
        f"Saved {len(chunks)} chunks to data/processed/chunks.json\n"
        f"Saved {len(triples)} raw triples to data/processed/triples.json\n"
        f"Saved cost report to outputs/cost_report.json\n"
        f"Extraction mode: {usage.extraction_mode}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raw triples from the curated tech corpus.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use curated deterministic triples even when OPENAI_API_KEY is set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(force_offline=args.offline)
