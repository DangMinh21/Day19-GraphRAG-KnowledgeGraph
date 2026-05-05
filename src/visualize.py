"""Visualize the NetworkX knowledge graph with Matplotlib."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import KNOWLEDGE_GRAPH_IMAGE_PATH, PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt
import networkx as nx

from src.graph_store import build_graph, load_normalized_triples


NODE_COLORS = {
    "company": "#4C78A8",
    "person": "#F58518",
    "product": "#54A24B",
    "location": "#B279A2",
    "year": "#E45756",
    "entity": "#8D8D8D",
}

NODE_SIZES = {
    "company": 1250,
    "person": 700,
    "product": 800,
    "location": 760,
    "year": 620,
    "entity": 650,
}


def collapse_multidigraph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse parallel edges and keep combined relation labels."""

    collapsed = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        collapsed.add_node(node, **data)

    for source, target, data in graph.edges(data=True):
        relation = data.get("relation", "RELATED_TO")
        if collapsed.has_edge(source, target):
            relations = set(collapsed[source][target]["relation"].split(", "))
            relations.add(relation)
            collapsed[source][target]["relation"] = ", ".join(sorted(relations))
        else:
            collapsed.add_edge(source, target, relation=relation)

    return collapsed


def filter_graph_for_readability(
    graph: nx.MultiDiGraph,
    focus_nodes: list[str],
    max_hops: int = 1,
) -> nx.MultiDiGraph:
    """Return a subgraph around important companies for a readable lab image."""

    selected_nodes: set[str] = set()
    undirected = graph.to_undirected()

    for focus_node in focus_nodes:
        if focus_node not in graph:
            continue
        lengths = nx.single_source_shortest_path_length(undirected, focus_node, cutoff=max_hops)
        selected_nodes.update(lengths)

    if not selected_nodes:
        return graph

    return graph.subgraph(sorted(selected_nodes)).copy()


def draw_graph(
    graph: nx.MultiDiGraph,
    output_path: Path = KNOWLEDGE_GRAPH_IMAGE_PATH,
    title: str = "Tech Company Knowledge Graph",
) -> None:
    """Draw and save the graph image."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph_to_draw = collapse_multidigraph(graph)

    plt.figure(figsize=(22, 16))
    position = nx.spring_layout(graph_to_draw, seed=42, k=1.2, iterations=180)

    node_types = nx.get_node_attributes(graph_to_draw, "type")
    node_colors = [NODE_COLORS.get(node_types.get(node, "entity"), "#8D8D8D") for node in graph_to_draw.nodes]
    node_sizes = [NODE_SIZES.get(node_types.get(node, "entity"), 650) for node in graph_to_draw.nodes]

    nx.draw_networkx_edges(
        graph_to_draw,
        position,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        edge_color="#BBBBBB",
        width=1.2,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph_to_draw,
        position,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#222222",
        linewidths=0.7,
        alpha=0.95,
    )
    nx.draw_networkx_labels(
        graph_to_draw,
        position,
        font_size=8,
        font_color="#111111",
        font_weight="bold",
    )

    edge_labels = {
        (source, target): data["relation"]
        for source, target, data in graph_to_draw.edges(data=True)
        if data["relation"]
        in {
            "ACQUIRED",
            "INVESTED_IN",
            "PARENT_COMPANY_OF",
            "SUBSIDIARY_OF",
            "PARTNERED_WITH",
            "PROVIDES_INFRASTRUCTURE_FOR",
        }
    }
    nx.draw_networkx_edge_labels(
        graph_to_draw,
        position,
        edge_labels=edge_labels,
        font_size=6,
        font_color="#444444",
        rotate=False,
        label_pos=0.55,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="#222222",
            markersize=10,
            label=node_type.title(),
        )
        for node_type, color in NODE_COLORS.items()
    ]
    plt.legend(handles=legend_handles, loc="lower left", frameon=True)
    plt.title(title, fontsize=18, pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def run_visualization(full_graph: bool = False) -> None:
    """Load normalized triples and save the graph visualization."""

    triples = load_normalized_triples()
    graph = build_graph(triples)

    if full_graph:
        graph_to_plot = graph
        title = "Tech Company Knowledge Graph"
    else:
        graph_to_plot = filter_graph_for_readability(
            graph,
            focus_nodes=["OpenAI", "Microsoft", "Google", "Meta", "Amazon", "NVIDIA", "Tesla"],
            max_hops=1,
        )
        title = "Tech Company Knowledge Graph: Key Company Neighborhoods"

    draw_graph(graph_to_plot, title=title)
    print(
        f"Saved graph image to {KNOWLEDGE_GRAPH_IMAGE_PATH}\n"
        f"Rendered nodes: {graph_to_plot.number_of_nodes()}\n"
        f"Rendered edges: {graph_to_plot.number_of_edges()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the NetworkX knowledge graph.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Render the full graph instead of a readable focused subgraph.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visualization(full_graph=args.full)
