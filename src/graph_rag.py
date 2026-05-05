"""GraphRAG querying over the normalized NetworkX knowledge graph."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import networkx as nx

from src.config import OPENAI_ANSWER_MODEL, OPENAI_API_KEY
from src.graph_store import (
    NormalizedTriple,
    build_graph,
    canonicalize_entity,
    load_normalized_triples,
)


@dataclass(frozen=True)
class SupportingTriple:
    """A graph triple used as evidence for an answer."""

    subject: str
    relation: str
    object: str
    source_chunk_id: str
    confidence: float
    distance: int

    def to_sentence(self) -> str:
        relation_text = self.relation.lower().replace("_", " ")
        return f"{self.subject} {relation_text} {self.object}."

    def to_context_line(self) -> str:
        return (
            f"{self.subject} --{self.relation}--> {self.object} "
            f"({self.source_chunk_id}, confidence={self.confidence:.2f}, distance={self.distance})"
        )


@dataclass(frozen=True)
class GraphRAGResult:
    """GraphRAG answer plus matched entities and supporting triples."""

    answer: str
    matched_entities: list[str]
    supporting_triples: list[SupportingTriple]

    @property
    def context_text(self) -> str:
        return "\n".join(triple.to_context_line() for triple in self.supporting_triples)


class GraphRAG:
    """Entity-linked graph traversal and answer generation."""

    def __init__(self, triples: list[NormalizedTriple]) -> None:
        self.triples = triples
        self.graph = build_graph(triples)
        self.nodes_by_casefold = {node.casefold(): node for node in self.graph.nodes}

    @classmethod
    def from_saved_graph(cls) -> "GraphRAG":
        return cls(load_normalized_triples())

    def match_entities(self, question: str) -> list[str]:
        """Match question mentions to graph nodes using simple deterministic rules."""

        question_clean = " ".join(question.strip().split())
        question_lower = question_clean.casefold()
        matches: dict[str, int] = {}

        # Exact canonical alias match first.
        for token in extract_capitalized_phrases(question_clean):
            canonical = canonicalize_entity(token)
            node = self.nodes_by_casefold.get(canonical.casefold())
            if node:
                matches[node] = min(matches.get(node, len(question_clean)), question_lower.find(token.casefold()))

        # Then allow substring matches for multi-word entities already in graph.
        for node in sorted(self.graph.nodes, key=len, reverse=True):
            node_lower = node.casefold()
            if len(node) < 3 or node in matches:
                continue
            found = re.search(rf"\b{re.escape(node_lower)}\b", question_lower)
            if found:
                matches[node] = found.start()

        return [node for node, _ in sorted(matches.items(), key=lambda item: item[1])]

    def collect_supporting_triples(
        self,
        matched_entities: list[str],
        max_hops: int = 2,
        max_triples: int = 18,
    ) -> list[SupportingTriple]:
        """Collect graph triples reachable within `max_hops` of matched entities."""

        if not matched_entities:
            return []

        undirected = self.graph.to_undirected()
        node_distance: dict[str, int] = {}

        for entity in matched_entities:
            distances = nx.single_source_shortest_path_length(undirected, entity, cutoff=max_hops)
            for node, distance in distances.items():
                node_distance[node] = min(node_distance.get(node, distance), distance)

        supporting: list[SupportingTriple] = []
        seen_edges: set[tuple[str, str, str]] = set()
        selected_nodes = set(node_distance)

        for source, target, data in self.graph.edges(data=True):
            if source not in selected_nodes or target not in selected_nodes:
                continue

            relation = data.get("relation", "RELATED_TO")
            key = (source, relation, target)
            if key in seen_edges:
                continue

            distance = min(node_distance[source], node_distance[target])
            supporting.append(
                SupportingTriple(
                    subject=source,
                    relation=relation,
                    object=target,
                    source_chunk_id=data.get("source_chunk_id", "unknown"),
                    confidence=float(data.get("confidence", 0.0)),
                    distance=distance,
                )
            )
            seen_edges.add(key)

        supporting.sort(key=lambda triple: (triple.distance, -triple.confidence, triple.subject, triple.object))
        return supporting[:max_triples]

    def answer(
        self,
        question: str,
        max_hops: int = 2,
        max_triples: int = 18,
        use_openai: bool = False,
    ) -> GraphRAGResult:
        """Answer a question from graph context."""

        matched_entities = self.match_entities(question)
        supporting_triples = self.collect_supporting_triples(
            matched_entities,
            max_hops=max_hops,
            max_triples=max_triples,
        )

        if use_openai and OPENAI_API_KEY:
            answer = generate_openai_answer(question, supporting_triples)
        else:
            answer = generate_offline_answer(question, matched_entities, supporting_triples)

        return GraphRAGResult(
            answer=answer,
            matched_entities=matched_entities,
            supporting_triples=supporting_triples,
        )


def extract_capitalized_phrases(text: str) -> list[str]:
    """Extract likely entity mentions from capitalized words in a question."""

    phrases = re.findall(r"\b(?:[A-Z][A-Za-z0-9-]*)(?:\s+[A-Z][A-Za-z0-9-]*)*\b", text)
    return [phrase.strip() for phrase in phrases if phrase.strip()]


def tokenize_graph_text(text: str) -> set[str]:
    """Small local tokenizer for scoring graph evidence."""

    stop_words = {
        "and",
        "are",
        "did",
        "does",
        "for",
        "from",
        "into",
        "the",
        "through",
        "what",
        "where",
        "which",
        "with",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop_words
    }


def infer_requested_relation(question: str) -> str | None:
    """Infer the main relation requested by a short natural-language question."""

    question_lower = question.casefold()
    if "acquire" in question_lower:
        return "ACQUIRED"
    if "subsidiary" in question_lower or "belong" in question_lower:
        return "SUBSIDIARY_OF"
    if "create" in question_lower or "product" in question_lower:
        return "CREATED_PRODUCT"
    if "found" in question_lower:
        return "FOUNDED_BY"
    if "invest" in question_lower:
        return "INVESTED_IN"
    return None


def answer_yes_no_question(
    question: str,
    matched_entities: list[str],
    supporting_triples: list[SupportingTriple],
) -> str | None:
    """Answer common adversarial yes/no relation questions from graph triples."""

    if not re.match(r"^\s*(did|does|is|was)\b", question.casefold()) or len(matched_entities) < 2:
        return None

    relation = infer_requested_relation(question)
    if relation is None:
        return None

    subject, object_ = matched_entities[0], matched_entities[1]
    triples = supporting_triples

    if relation == "SUBSIDIARY_OF":
        direct_match = any(
            triple.subject == subject
            and triple.object == object_
            and triple.relation in {"SUBSIDIARY_OF", "PARENT_COMPANY_OF"}
            for triple in triples
        )
        if direct_match:
            return f"Yes. {subject} is connected to {object_} by a subsidiary or parent-company relation."

        true_owner = next(
            (triple.object for triple in triples if triple.subject == subject and triple.relation == "SUBSIDIARY_OF"),
            None,
        )
        true_child = next(
            (triple.object for triple in triples if triple.subject == object_ and triple.relation == "PARENT_COMPANY_OF"),
            None,
        )
        evidence = f" {subject} is a subsidiary of {true_owner}." if true_owner else ""
        if true_child:
            evidence += f" {object_} is parent company of {true_child}."
        creator = next(
            (
                triple.subject
                for triple in triples
                if triple.object == subject and triple.relation == "CREATED_PRODUCT"
            ),
            None,
        )
        if creator:
            evidence += f" {subject} was created by {creator}."
        return f"No.{evidence}".strip()

    direct_match = any(
        triple.subject == subject and triple.object == object_ and triple.relation == relation
        for triple in triples
    )
    if direct_match:
        relation_text = relation.lower().replace("_", " ")
        return f"Yes. The graph contains: {subject} {relation_text} {object_}."

    if relation == "FOUNDED_BY":
        founders = [
            triple.object
            for triple in triples
            if triple.subject == subject and triple.relation == "FOUNDED_BY"
        ]
        if founders:
            return f"No. {subject} was founded by {', '.join(founders)}."

    if relation == "CREATED_PRODUCT":
        creator = next(
            (
                triple.subject
                for triple in triples
                if triple.object == object_ and triple.relation == "CREATED_PRODUCT"
            ),
            None,
        )
        if creator:
            return f"No. {creator} created {object_}."

    alternate = next(
        (
            triple
            for triple in triples
            if triple.object == object_
            and triple.relation in {relation, "ACQUIRED_BY"}
            and triple.subject != subject
        ),
        None,
    )
    if alternate:
        return f"No. The graph shows: {alternate.to_sentence()}"

    reverse = next(
        (
            triple
            for triple in triples
            if triple.subject == object_
            and triple.object == subject
            and triple.relation in {"ACQUIRED_BY", "SUBSIDIARY_OF"}
        ),
        None,
    )
    if reverse:
        return f"No. The graph shows: {reverse.to_sentence()}"

    return f"No. The graph does not contain a {relation} edge from {subject} to {object_}."


def generate_offline_answer(
    question: str,
    matched_entities: list[str],
    supporting_triples: list[SupportingTriple],
) -> str:
    """Create a deterministic answer from graph evidence."""

    if not matched_entities:
        return "No graph entity was matched for this question."
    if not supporting_triples:
        return f"Matched entities: {', '.join(matched_entities)}, but no supporting triples were found."

    yes_no_answer = answer_yes_no_question(question, matched_entities, supporting_triples)
    if yes_no_answer is not None:
        return yes_no_answer

    query_tokens = tokenize_graph_text(question)
    scored: list[tuple[int, SupportingTriple]] = []

    for triple in supporting_triples:
        triple_text = f"{triple.subject} {triple.relation} {triple.object}"
        overlap = len(query_tokens & tokenize_graph_text(triple_text))
        entity_bonus = int(triple.subject in matched_entities) + int(triple.object in matched_entities)
        scored.append((overlap + entity_bonus - triple.distance, triple))

    selected: list[SupportingTriple] = []
    seen: set[tuple[str, str, str]] = set()
    for _, triple in sorted(scored, key=lambda item: item[0], reverse=True):
        key = (triple.subject, triple.relation, triple.object)
        if key in seen:
            continue
        selected.append(triple)
        seen.add(key)
        if len(selected) == 5:
            break

    evidence = " ".join(triple.to_sentence() for triple in selected)
    return f"Graph evidence: {evidence}"


def generate_openai_answer(question: str, supporting_triples: list[SupportingTriple]) -> str:
    """Generate a concise answer with OpenAI from graph triples."""

    from openai import OpenAI

    context = "\n".join(triple.to_context_line() for triple in supporting_triples)
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_ANSWER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided knowledge graph triples. "
                    "Mention the key supporting relationships briefly."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nGraph triples:\n{context}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GraphRAG over the NetworkX graph.")
    parser.add_argument("question", help="Question to answer.")
    parser.add_argument("--max-hops", type=int, default=2, help="Traversal depth around matched entities.")
    parser.add_argument("--max-triples", type=int, default=18, help="Maximum supporting triples to include.")
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI for answer generation when OPENAI_API_KEY is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag = GraphRAG.from_saved_graph()
    result = rag.answer(
        args.question,
        max_hops=args.max_hops,
        max_triples=args.max_triples,
        use_openai=args.openai,
    )
    print("Answer:")
    print(result.answer)
    print("\nMatched entities:")
    print(", ".join(result.matched_entities) or "None")
    print("\nSupporting triples:")
    print(result.context_text)


if __name__ == "__main__":
    main()
