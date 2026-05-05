"""Simple Flat RAG baseline using TF-IDF retrieval over corpus chunks."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import OPENAI_ANSWER_MODEL, OPENAI_API_KEY
from src.corpus import CorpusChunk, chunk_corpus, load_chunks


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved corpus chunk with its TF-IDF similarity score."""

    chunk_id: str
    text: str
    score: float


@dataclass(frozen=True)
class FlatRAGResult:
    """Flat RAG answer plus retrieved evidence."""

    answer: str
    retrieved_chunks: list[RetrievedChunk]

    @property
    def context_text(self) -> str:
        return "\n".join(
            f"{chunk.chunk_id} | score={chunk.score:.3f} | {chunk.text}"
            for chunk in self.retrieved_chunks
        )


class FlatRAG:
    """A small TF-IDF retriever and answer generator baseline."""

    def __init__(self, chunks: list[CorpusChunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.chunk_matrix = self.vectorizer.fit_transform(chunk.text for chunk in chunks)

    @classmethod
    def from_saved_chunks(cls) -> "FlatRAG":
        """Load `data/processed/chunks.json`, or rebuild chunks if needed."""

        try:
            chunks = load_chunks()
        except FileNotFoundError:
            chunks = chunk_corpus()
        return cls(chunks)

    def retrieve(self, question: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Retrieve the most similar chunks for a question."""

        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.chunk_matrix).flatten()
        ranked_indexes = scores.argsort()[::-1][:top_k]

        return [
            RetrievedChunk(
                chunk_id=self.chunks[index].chunk_id,
                text=self.chunks[index].text,
                score=float(scores[index]),
            )
            for index in ranked_indexes
            if scores[index] > 0
        ]

    def answer(self, question: str, top_k: int = 4, use_openai: bool = False) -> FlatRAGResult:
        """Answer a question with retrieved chunk context."""

        retrieved_chunks = self.retrieve(question, top_k=top_k)
        if use_openai and OPENAI_API_KEY:
            answer = generate_openai_answer(question, retrieved_chunks)
        else:
            answer = generate_offline_answer(question, retrieved_chunks)
        return FlatRAGResult(answer=answer, retrieved_chunks=retrieved_chunks)


def tokenize_query(text: str) -> set[str]:
    """Tokenize query text for the offline sentence selector."""

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "by",
        "did",
        "does",
        "for",
        "in",
        "is",
        "of",
        "or",
        "the",
        "through",
        "to",
        "what",
        "where",
        "which",
        "who",
        "with",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop_words
    }


def split_sentences(text: str) -> list[str]:
    """Split chunk text into simple sentences."""

    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def generate_offline_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Create a deterministic baseline answer from retrieved context.

    The answer is intentionally extractive. This keeps Flat RAG easy to inspect
    and makes hallucinations or missed relationships visible during evaluation.
    """

    if not retrieved_chunks:
        return "No relevant context was retrieved."

    top_score = retrieved_chunks[0].score
    if len(retrieved_chunks) == 1 or top_score >= retrieved_chunks[1].score * 2:
        candidate_chunks = retrieved_chunks[:1]
    else:
        candidate_chunks = retrieved_chunks

    query_tokens = tokenize_query(question)
    scored_sentences: list[tuple[int, str]] = []

    for chunk in candidate_chunks:
        for sentence in split_sentences(chunk.text):
            sentence_tokens = tokenize_query(sentence)
            overlap = len(query_tokens & sentence_tokens)
            if overlap > 0:
                scored_sentences.append((overlap, sentence))

    if not scored_sentences:
        return retrieved_chunks[0].text

    unique_sentences: list[str] = []
    seen: set[str] = set()
    for _, sentence in sorted(scored_sentences, key=lambda item: item[0], reverse=True):
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
        if len(unique_sentences) == 3:
            break

    return " ".join(unique_sentences)


def generate_openai_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Generate a concise answer with OpenAI using retrieved chunk context."""

    from openai import OpenAI

    context = "\n".join(f"{chunk.chunk_id}: {chunk.text}" for chunk in retrieved_chunks)
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_ANSWER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. If the context is insufficient, "
                    "say that the answer is not supported by the retrieved chunks."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Flat RAG TF-IDF baseline.")
    parser.add_argument("question", help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI for answer generation when OPENAI_API_KEY is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag = FlatRAG.from_saved_chunks()
    result = rag.answer(args.question, top_k=args.top_k, use_openai=args.openai)
    print("Answer:")
    print(result.answer)
    print("\nRetrieved context:")
    print(result.context_text)


if __name__ == "__main__":
    main()
