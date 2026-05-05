"""End-to-end runner for the Day 19 GraphRAG lab submission.

Teachers can run this file from the repository root to regenerate all required
artifacts and see the pipeline status directly in the terminal.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.config import (
    BENCHMARK_REPORT_PATH,
    BENCHMARK_RESULTS_PATH,
    CHUNKS_PATH,
    COST_REPORT_PATH,
    KNOWLEDGE_GRAPH_IMAGE_PATH,
    NORMALIZED_TRIPLES_PATH,
    RAW_CORPUS_PATH,
    SUBMISSION_STATUS_JSON_PATH,
    SUBMISSION_STATUS_PATH,
    TRIPLES_PATH,
)
from src.evaluate import evaluate, save_markdown_report, save_results
from src.extract_triples import run_extraction
from src.graph_store import build_graph, graph_summary, load_normalized_triples, run_normalization
from src.visualize import run_visualization


@dataclass
class StepResult:
    """Runtime metadata for one pipeline step."""

    name: str
    status: str
    duration_seconds: float
    outputs: list[str]
    details: dict[str, object]


def hrule() -> None:
    print("=" * 88)


def log_phase(title: str, inputs: list[Path], process: str, outputs: list[Path]) -> None:
    """Print a readable phase banner for terminal-based grading."""

    hrule()
    print(f"PHASE: {title}")
    print("Input:")
    for path in inputs:
        print(f"  - {path}")
    print("Process:")
    print(f"  - {process}")
    print("Expected output:")
    for path in outputs:
        print(f"  - {path}")
    hrule()


def run_step(
    *,
    name: str,
    inputs: list[Path],
    process: str,
    outputs: list[Path],
    action: Callable[[], dict[str, object]],
) -> StepResult:
    """Run one pipeline phase with logging and timing."""

    log_phase(name, inputs, process, outputs)
    start = time.perf_counter()

    try:
        details = action()
        duration = time.perf_counter() - start
        missing_outputs = [str(path) for path in outputs if not path.exists()]
        status = "OK" if not missing_outputs else "MISSING_OUTPUT"
        if missing_outputs:
            details["missing_outputs"] = missing_outputs
        print(f"Status: {status}")
        print(f"Duration: {duration:.2f}s")
        return StepResult(
            name=name,
            status=status,
            duration_seconds=round(duration, 3),
            outputs=[str(path) for path in outputs],
            details=details,
        )
    except Exception as exc:
        duration = time.perf_counter() - start
        print(f"Status: FAILED")
        print(f"Duration: {duration:.2f}s")
        print(f"Error: {exc}")
        return StepResult(
            name=name,
            status="FAILED",
            duration_seconds=round(duration, 3),
            outputs=[str(path) for path in outputs],
            details={"error": repr(exc)},
        )


def file_size(path: Path) -> str:
    """Return a compact human-readable file size."""

    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def build_artifact_summary(paths: list[Path]) -> list[dict[str, object]]:
    """Collect existence and size metadata for submission artifacts."""

    return [
        {
            "path": str(path),
            "exists": path.exists(),
            "size": file_size(path),
        }
        for path in paths
    ]


def write_submission_status(
    *,
    step_results: list[StepResult],
    artifact_paths: list[Path],
    benchmark_summary: dict[str, object],
    total_duration_seconds: float,
) -> None:
    """Write Markdown and JSON status reports for graders."""

    def make_payload(artifacts: list[dict[str, object]]) -> dict[str, object]:
        all_ok = all(result.status == "OK" for result in step_results) and all(
            artifact["exists"] for artifact in artifacts
        )
        return {
            "overall_status": "OK" if all_ok else "CHECK_REQUIRED",
            "total_duration_seconds": round(total_duration_seconds, 3),
            "benchmark_summary": benchmark_summary,
            "steps": [result.__dict__ for result in step_results],
            "artifacts": artifacts,
        }

    def render_markdown(status_payload: dict[str, object]) -> str:
        lines = [
            "# Submission Status",
            "",
            f"- Overall status: {status_payload['overall_status']}",
            f"- Total runtime: {total_duration_seconds:.2f}s",
            f"- Flat RAG accuracy: {benchmark_summary.get('flat_accuracy', 'n/a')}",
            f"- GraphRAG accuracy: {benchmark_summary.get('graph_accuracy', 'n/a')}",
            "",
            "## Pipeline Steps",
            "",
            "| Step | Status | Duration |",
            "|---|---|---:|",
        ]

        for result in step_results:
            lines.append(f"| {result.name} | {result.status} | {result.duration_seconds:.2f}s |")

        lines.extend(
            [
                "",
                "## Submission Artifacts",
                "",
                "| Artifact | Exists | Size |",
                "|---|---:|---:|",
            ]
        )

        for artifact in status_payload["artifacts"]:
            exists = "yes" if artifact["exists"] else "no"
            lines.append(f"| `{artifact['path']}` | {exists} | {artifact['size']} |")

        lines.extend(
            [
                "",
                "## Notes For Grading",
                "",
                "- `python main.py` regenerates all deterministic offline artifacts.",
                "- `python main.py --openai` uses OpenAI for extraction and answer generation when `OPENAI_API_KEY` is configured.",
                "- Benchmark CSV is the raw output; benchmark Markdown is the readable report.",
                "",
            ]
        )
        return "\n".join(lines)

    SUBMISSION_STATUS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    # First write creates the status files; second write reports their real sizes.
    for _ in range(2):
        status_payload = make_payload(build_artifact_summary(artifact_paths))
        SUBMISSION_STATUS_JSON_PATH.write_text(
            json.dumps(status_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        SUBMISSION_STATUS_PATH.write_text(render_markdown(status_payload), encoding="utf-8")


def run_pipeline(use_openai: bool = False, full_graph: bool = False) -> int:
    """Run the complete submission pipeline."""

    total_start = time.perf_counter()
    mode = "OpenAI API" if use_openai else "offline deterministic"
    print("\nDay 19 GraphRAG End-to-End Pipeline")
    print(f"Run mode: {mode}")
    print(f"Graph visualization: {'full graph' if full_graph else 'focused graph'}")
    print()

    step_results: list[StepResult] = []

    step_results.append(
        run_step(
            name="1. Corpus chunking and triple extraction",
            inputs=[RAW_CORPUS_PATH],
            process=(
                "Load curated Markdown corpus, split into chunks, extract raw triples, "
                "and write token/time cost report."
            ),
            outputs=[CHUNKS_PATH, TRIPLES_PATH, COST_REPORT_PATH],
            action=lambda: extraction_action(use_openai=use_openai),
        )
    )

    step_results.append(
        run_step(
            name="2. Triple normalization and NetworkX graph build",
            inputs=[TRIPLES_PATH],
            process="Normalize aliases, deduplicate triples, build directed NetworkX graph, and summarize nodes/edges.",
            outputs=[NORMALIZED_TRIPLES_PATH],
            action=normalization_action,
        )
    )

    step_results.append(
        run_step(
            name="3. Knowledge graph visualization",
            inputs=[NORMALIZED_TRIPLES_PATH],
            process="Render the knowledge graph with typed node colors and relation labels.",
            outputs=[KNOWLEDGE_GRAPH_IMAGE_PATH],
            action=lambda: visualization_action(full_graph=full_graph),
        )
    )

    benchmark_holder: dict[str, object] = {}
    step_results.append(
        run_step(
            name="4. Flat RAG vs GraphRAG benchmark",
            inputs=[CHUNKS_PATH, NORMALIZED_TRIPLES_PATH],
            process="Run both Flat RAG and GraphRAG on 20 benchmark questions, then export CSV and Markdown report.",
            outputs=[BENCHMARK_RESULTS_PATH, BENCHMARK_REPORT_PATH],
            action=lambda: benchmark_action(use_openai=use_openai, benchmark_holder=benchmark_holder),
        )
    )

    total_duration = time.perf_counter() - total_start
    artifact_paths = [
        CHUNKS_PATH,
        TRIPLES_PATH,
        NORMALIZED_TRIPLES_PATH,
        COST_REPORT_PATH,
        KNOWLEDGE_GRAPH_IMAGE_PATH,
        BENCHMARK_RESULTS_PATH,
        BENCHMARK_REPORT_PATH,
        SUBMISSION_STATUS_PATH,
        SUBMISSION_STATUS_JSON_PATH,
    ]

    write_submission_status(
        step_results=step_results,
        artifact_paths=artifact_paths,
        benchmark_summary=benchmark_holder,
        total_duration_seconds=total_duration,
    )

    hrule()
    print("FINAL SUBMISSION STATUS")
    print(f"Status report: {SUBMISSION_STATUS_PATH}")
    print(f"Status JSON:   {SUBMISSION_STATUS_JSON_PATH}")
    print("Artifacts:")
    for artifact in build_artifact_summary(artifact_paths):
        print(f"  - {artifact['path']} | exists={artifact['exists']} | size={artifact['size']}")
    print(f"Total runtime: {total_duration:.2f}s")
    hrule()

    failed = [result for result in step_results if result.status != "OK"]
    return 1 if failed else 0


def extraction_action(use_openai: bool) -> dict[str, object]:
    """Run extraction and return artifact counts."""

    run_extraction(force_offline=not use_openai)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    triples = json.loads(TRIPLES_PATH.read_text(encoding="utf-8"))
    cost_report = json.loads(COST_REPORT_PATH.read_text(encoding="utf-8"))
    return {
        "num_chunks": len(chunks),
        "num_raw_triples": len(triples),
        "extraction_mode": cost_report.get("extraction_mode"),
        "estimated_cost_usd": cost_report.get("estimated_cost_usd"),
    }


def normalization_action() -> dict[str, object]:
    """Run normalization and return graph summary."""

    run_normalization()
    triples = load_normalized_triples()
    graph = build_graph(triples)
    return graph_summary(graph)


def visualization_action(full_graph: bool) -> dict[str, object]:
    """Run visualization and return image metadata."""

    run_visualization(full_graph=full_graph)
    return {
        "image_path": str(KNOWLEDGE_GRAPH_IMAGE_PATH),
        "image_size": file_size(KNOWLEDGE_GRAPH_IMAGE_PATH),
    }


def benchmark_action(use_openai: bool, benchmark_holder: dict[str, object]) -> dict[str, object]:
    """Run benchmark, save reports, and return accuracy summary."""

    df = evaluate(use_openai=use_openai)
    save_results(df)
    save_markdown_report(df)

    total = len(df)
    flat_correct = int(df["flat_rag_correct"].sum())
    graph_correct = int(df["graph_rag_correct"].sum())
    improvements = int(((~df["flat_rag_correct"]) & (df["graph_rag_correct"])).sum())

    summary = {
        "total_questions": total,
        "flat_correct": flat_correct,
        "graph_correct": graph_correct,
        "flat_accuracy": f"{flat_correct}/{total} = {flat_correct / total:.1%}",
        "graph_accuracy": f"{graph_correct}/{total} = {graph_correct / total:.1%}",
        "graph_only_wins": improvements,
    }
    benchmark_holder.update(summary)

    print(f"Flat RAG accuracy: {summary['flat_accuracy']}")
    print(f"GraphRAG accuracy: {summary['graph_accuracy']}")
    print(f"GraphRAG-only wins: {improvements}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete Day 19 GraphRAG submission pipeline.")
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI extraction and answer generation when OPENAI_API_KEY is configured.",
    )
    parser.add_argument(
        "--full-graph",
        action="store_true",
        help="Render the full knowledge graph image instead of the focused readable graph.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_pipeline(use_openai=args.openai, full_graph=args.full_graph)


if __name__ == "__main__":
    raise SystemExit(main())
