from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import GraphEdge, GraphNode, SourceSnippet, TokenMeasurement
from app.services.storage import LocalStorage
from app.services.token_service import TokenService


@dataclass
class GraphRetrievalResult:
    context: str
    snippets: list[SourceSnippet]
    selected_nodes: list[GraphNode]
    selected_edges: list[GraphEdge]
    token_measurement: TokenMeasurement


class GraphRetrievalService:
    def __init__(self, storage: LocalStorage, token_service: TokenService) -> None:
        self.storage = storage
        self.token_service = token_service

    def build_context(self, repo_id: str, query: str, max_nodes: int = 24) -> GraphRetrievalResult:
        codegraph = self.storage.load_codegraph(repo_id)
        graphify = self.storage.load_graphify(repo_id)
        if codegraph is None:
            raise ValueError("CodeGraph output not found for repo.")

        all_nodes = codegraph.nodes + (graphify.nodes if graphify else [])
        all_edges = codegraph.edges + (graphify.edges if graphify else [])
        terms = self._terms(query)
        scores = [(self._score_node(node, terms), node) for node in all_nodes]
        anchors = [node for score, node in sorted(scores, key=lambda item: item[0], reverse=True) if score > 0][:10]

        selected: dict[str, GraphNode] = {node.node_id: node for node in anchors}
        adjacent_edges: list[GraphEdge] = []
        for edge in all_edges:
            if edge.source_node in selected or edge.target_node in selected:
                adjacent_edges.append(edge)
                for node in all_nodes:
                    if node.node_id in {edge.source_node, edge.target_node}:
                        selected.setdefault(node.node_id, node)
            if len(selected) >= max_nodes:
                break

        if not selected:
            selected = {node.node_id: node for _, node in scores[: min(max_nodes, len(scores))]}

        selected_nodes = list(selected.values())[:max_nodes]
        selected_ids = {node.node_id for node in selected_nodes}
        selected_edges = [
            edge for edge in adjacent_edges if edge.source_node in selected_ids and edge.target_node in selected_ids
        ][: max_nodes * 2]

        snippets = self._snippets(selected_nodes)
        context = self._format_context(selected_nodes, selected_edges, snippets)
        measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", context)
        return GraphRetrievalResult(
            context=context,
            snippets=snippets,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            token_measurement=measurement,
        )

    def _terms(self, query: str) -> list[str]:
        return [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", query)]

    def _score_node(self, node: GraphNode, terms: list[str]) -> float:
        haystack = " ".join(
            filter(
                None,
                [
                    node.label,
                    node.node_type,
                    node.file_path or "",
                    node.source_snippet or "",
                    " ".join(str(value) for value in node.metadata.values()),
                ],
            )
        ).lower()
        score = 0.0
        for term in terms:
            if term in node.label.lower():
                score += 8
            if term in haystack:
                score += haystack.count(term)
        if node.node_type in {"function", "class", "method"}:
            score += 0.2
        return score

    def _snippets(self, nodes: list[GraphNode]) -> list[SourceSnippet]:
        snippets: list[SourceSnippet] = []
        seen: set[tuple[str, int | None, int | None]] = set()
        for node in nodes:
            if not node.file_path or not node.source_snippet:
                continue
            key = (node.file_path, node.line_start, node.line_end)
            if key in seen:
                continue
            seen.add(key)
            snippets.append(
                SourceSnippet(
                    file_path=node.file_path,
                    line_start=node.line_start or 1,
                    line_end=node.line_end or node.line_start or 1,
                    text=node.source_snippet,
                    source="graph",
                )
            )
        return snippets

    def _format_context(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        snippets: list[SourceSnippet],
    ) -> str:
        node_lines = [
            f"- {node.node_id} [{node.node_type}] {node.label} ({node.file_path or 'external'}:{node.line_start or '-'})"
            for node in nodes
        ]
        edge_lines = [
            f"- {edge.source_node} --{edge.edge_type}--> {edge.target_node}"
            for edge in edges
        ]
        snippet_lines = [
            f"### {snippet.file_path}:{snippet.line_start}-{snippet.line_end}\n{snippet.text}"
            for snippet in snippets
        ]
        return "\n".join(
            [
                "Graph-selected nodes:",
                *node_lines,
                "",
                "Graph relationships:",
                *edge_lines,
                "",
                "Source snippets:",
                *snippet_lines,
            ]
        )

