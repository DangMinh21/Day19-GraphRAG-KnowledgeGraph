"""Benchmark Flat RAG against GraphRAG on the lab question set."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import BENCHMARK_QUESTIONS_PATH, BENCHMARK_REPORT_PATH, BENCHMARK_RESULTS_PATH
from src.flat_rag import FlatRAG
from src.graph_rag import GraphRAG


@dataclass(frozen=True)
class BenchmarkQuestion:
    """A benchmark item loaded from `data/benchmark/questions.json`."""

    id: str
    question: str
    expected_answer: str
    type: str
    required_entities: list[str]


def load_questions(path: Path = BENCHMARK_QUESTIONS_PATH) -> list[BenchmarkQuestion]:
    """Load benchmark questions from JSON."""

    records = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkQuestion(**record) for record in records]


def normalize_text(text: str) -> str:
    """Lowercase text and keep only alphanumeric tokens for matching."""

    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def entity_present(entity: str, text: str) -> bool:
    """Check whether an entity phrase appears in text after light normalization."""

    normalized_entity = normalize_text(entity)
    normalized_text = normalize_text(text)
    return normalized_entity in normalized_text


def answer_has_negative_intent(expected_answer: str) -> bool:
    """Detect benchmark answers that require a negative yes/no response."""

    return normalize_text(expected_answer).startswith("no ")


def score_answer(
    *,
    answer: str,
    evidence: str,
    expected_answer: str,
    required_entities: list[str],
) -> tuple[bool, str]:
    """Score an answer with transparent string-based rules.

    This is intentionally simple for the lab. It rewards answers that contain
    the expected answer or enough required entities in either the final answer
    or its retrieved/graph evidence.
    """

    answer_text = f"{answer}\n{evidence}"
    expected_negative = answer_has_negative_intent(expected_answer)

    if expected_negative and not normalize_text(answer).startswith("no "):
        return False, "Expected a negative answer, but answer did not start with No."

    if entity_present(expected_answer, answer):
        return True, "Expected answer phrase appears in answer."

    present_entities = [entity for entity in required_entities if entity_present(entity, answer_text)]
    if expected_negative:
        # Negative questions usually need the false entity plus the correcting entity.
        correct = len(present_entities) >= max(2, len(required_entities) - 1)
    else:
        correct = len(present_entities) >= max(1, len(required_entities) - 1)

    note = f"Matched required entities: {', '.join(present_entities) or 'none'}."
    return correct, note


def evaluate(use_openai: bool = False) -> pd.DataFrame:
    """Run Flat RAG and GraphRAG on all benchmark questions."""

    questions = load_questions()
    flat_rag = FlatRAG.from_saved_chunks()
    graph_rag = GraphRAG.from_saved_graph()
    rows: list[dict[str, object]] = []

    for item in questions:
        flat_result = flat_rag.answer(item.question, top_k=4, use_openai=use_openai)
        graph_result = graph_rag.answer(item.question, use_openai=use_openai)

        flat_correct, flat_note = score_answer(
            answer=flat_result.answer,
            evidence=flat_result.context_text,
            expected_answer=item.expected_answer,
            required_entities=item.required_entities,
        )
        graph_correct, graph_note = score_answer(
            answer=graph_result.answer,
            evidence=graph_result.context_text,
            expected_answer=item.expected_answer,
            required_entities=item.required_entities,
        )

        rows.append(
            {
                "question_id": item.id,
                "question": item.question,
                "question_type": item.type,
                "expected_answer": item.expected_answer,
                "flat_rag_answer": flat_result.answer,
                "graph_rag_answer": graph_result.answer,
                "flat_rag_correct": flat_correct,
                "graph_rag_correct": graph_correct,
                "flat_rag_context": flat_result.context_text,
                "graph_rag_context_triples": graph_result.context_text,
                "notes": f"Flat: {flat_note} Graph: {graph_note}",
            }
        )

    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, path: Path = BENCHMARK_RESULTS_PATH) -> None:
    """Save benchmark results to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def markdown_escape(text: object) -> str:
    """Escape table-sensitive characters for Markdown cells."""

    clean = str(text).replace("\n", "<br>")
    return clean.replace("|", "\\|")


def status_icon(value: bool) -> str:
    """Return a compact status label for Markdown tables."""

    return "OK" if bool(value) else "MISS"


def build_markdown_report(df: pd.DataFrame) -> str:
    """Build a human-readable benchmark report in Markdown."""

    total = len(df)
    flat_correct = int(df["flat_rag_correct"].sum())
    graph_correct = int(df["graph_rag_correct"].sum())
    improvements = df[(~df["flat_rag_correct"]) & (df["graph_rag_correct"])]

    lines: list[str] = [
        "# GraphRAG Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Total questions: {total}",
        f"- Flat RAG accuracy: {flat_correct}/{total} = {flat_correct / total:.1%}",
        f"- GraphRAG accuracy: {graph_correct}/{total} = {graph_correct / total:.1%}",
        f"- GraphRAG-only wins: {len(improvements)}",
        "",
        "## Accuracy By Question Type",
        "",
        "| Question Type | Flat RAG | GraphRAG |",
        "|---|---:|---:|",
    ]

    by_type = df.groupby("question_type")[["flat_rag_correct", "graph_rag_correct"]].mean()
    for question_type, row in by_type.iterrows():
        lines.append(
            f"| {markdown_escape(question_type)} | {row['flat_rag_correct']:.0%} | {row['graph_rag_correct']:.0%} |"
        )

    lines.extend(
        [
            "",
            "## Question Results",
            "",
            "| ID | Type | Question | Expected | Flat | Graph |",
            "|---|---|---|---|---|---|",
        ]
    )

    for _, row in df.iterrows():
        lines.append(
            "| "
            f"{markdown_escape(row['question_id'])} | "
            f"{markdown_escape(row['question_type'])} | "
            f"{markdown_escape(row['question'])} | "
            f"{markdown_escape(row['expected_answer'])} | "
            f"{status_icon(row['flat_rag_correct'])} | "
            f"{status_icon(row['graph_rag_correct'])} |"
        )

    lines.extend(
        [
            "",
            "## Where GraphRAG Improves",
            "",
        ]
    )

    if improvements.empty:
        lines.append("No GraphRAG-only wins were found in this run.")
    else:
        for _, row in improvements.iterrows():
            lines.extend(
                [
                    f"### {row['question_id']}: {row['question']}",
                    "",
                    f"- Expected: {row['expected_answer']}",
                    f"- Flat RAG: {row['flat_rag_answer']}",
                    f"- GraphRAG: {row['graph_rag_answer']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Notes",
            "",
            "- `OK` and `MISS` are produced by transparent string-based scoring rules in `src/evaluate.py`.",
            "- CSV remains the raw machine-readable output; this Markdown file is intended for review and lab reporting.",
            "",
        ]
    )

    return "\n".join(lines)


def save_markdown_report(df: pd.DataFrame, path: Path = BENCHMARK_REPORT_PATH) -> None:
    """Save a readable Markdown benchmark report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown_report(df), encoding="utf-8")


def print_summary(df: pd.DataFrame) -> None:
    """Print compact benchmark accuracy summary."""

    total = len(df)
    flat_correct = int(df["flat_rag_correct"].sum())
    graph_correct = int(df["graph_rag_correct"].sum())

    print(f"Saved benchmark results to {BENCHMARK_RESULTS_PATH}")
    print(f"Saved benchmark report to {BENCHMARK_REPORT_PATH}")
    print(f"Flat RAG accuracy: {flat_correct}/{total} = {flat_correct / total:.1%}")
    print(f"GraphRAG accuracy: {graph_correct}/{total} = {graph_correct / total:.1%}")
    print("\nAccuracy by question type:")
    by_type = df.groupby("question_type")[["flat_rag_correct", "graph_rag_correct"]].mean()
    print(by_type.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Flat RAG vs GraphRAG on benchmark questions.")
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI for answer generation when OPENAI_API_KEY is configured.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = evaluate(use_openai=args.openai)
    save_results(df)
    save_markdown_report(df)
    print_summary(df)


if __name__ == "__main__":
    main()
