"""Corpus loading and chunking utilities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.config import CHUNKS_PATH, RAW_CORPUS_PATH


@dataclass(frozen=True)
class CorpusChunk:
    """A small unit of corpus text used for extraction and retrieval."""

    chunk_id: str
    text: str
    source_path: str
    order: int


def load_markdown_corpus(path: Path = RAW_CORPUS_PATH) -> str:
    """Read the curated Markdown corpus from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    return path.read_text(encoding="utf-8")


def split_markdown_paragraphs(text: str) -> list[str]:
    """Split Markdown into compact paragraphs and skip headings."""

    paragraphs = re.split(r"\n\s*\n", text.strip())
    clean_paragraphs: list[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith("#"):
            continue
        clean_paragraphs.append(re.sub(r"\s+", " ", paragraph))
    return clean_paragraphs


def chunk_corpus(path: Path = RAW_CORPUS_PATH) -> list[CorpusChunk]:
    """Create paragraph-level chunks from the curated corpus."""

    text = load_markdown_corpus(path)
    paragraphs = split_markdown_paragraphs(text)
    return [
        CorpusChunk(
            chunk_id=f"chunk_{index:03d}",
            text=paragraph,
            source_path=str(path),
            order=index,
        )
        for index, paragraph in enumerate(paragraphs, start=1)
    ]


def save_chunks(chunks: Iterable[CorpusChunk], path: Path = CHUNKS_PATH) -> None:
    """Persist chunks as JSON for stable reruns and inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_chunks(path: Path = CHUNKS_PATH) -> list[CorpusChunk]:
    """Load previously saved chunks."""

    records = json.loads(path.read_text(encoding="utf-8"))
    return [CorpusChunk(**record) for record in records]

