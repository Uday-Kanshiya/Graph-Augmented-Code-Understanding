from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.schemas import GraphDocument, GraphEdge, GraphNode, SourceSnippet, TokenMeasurement
from app.services.file_utils import read_text_lossy
from app.services.storage import LocalStorage
from app.services.token_service import TokenService
from app.services.graphify_service import GraphifyService
from app.services.codegraph_service import CodeGraphService


@dataclass
class GraphRetrievalResult:
    context: str
    snippets: list[SourceSnippet]
    selected_nodes: list[GraphNode]
    selected_edges: list[GraphEdge]
    token_measurement: TokenMeasurement


class GraphRetrievalService:
    """Query-driven graph retrieval using Graphify and CodeGraph query engines.

    This implementation uses GraphifyService.query() and CodeGraphService.query()
    to execute semantic/structural queries against their respective graphs,
    then passes the results directly to the LLM for explanation.
    """

    def __init__(self, storage: LocalStorage, token_service: TokenService) -> None:
        self.storage = storage
        self.token_service = token_service
        self.graphify_service = GraphifyService(storage)
        self.codegraph_service = CodeGraphService()

    def build_context(self, repo_id: str, query: str, max_nodes: int = 8, source_selection: str = "merged") -> GraphRetrievalResult:
        graphify = self.storage.load_graphify(repo_id)
        codegraph = self.storage.load_codegraph(repo_id)

        if source_selection == "graphify":
            graph = graphify
        elif source_selection == "codegraph":
            graph = codegraph
        else:
            graph = graphify or codegraph

        if graph is None or not graph.nodes:
            if source_selection == "graphify":
                raise ValueError("Graphify output is not available for this repository.")
            if source_selection == "codegraph":
                raise ValueError("CodeGraph output is not available for this repository.")
            return self._fallback_context(repo_id)

        # Use the appropriate query service based on graph source
        if graph.source == "graphify":
            selected_nodes, selected_edges = self.graphify_service.query(graph, query, max_nodes)
        else:
            selected_nodes, selected_edges = self.codegraph_service.query(graph, query, max_nodes)

        snippets = [self._node_to_snippet(node) for node in selected_nodes if node.source_snippet]
        context_text = self._build_graph_context_text(query, selected_nodes, selected_edges)
        measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", context_text)

        return GraphRetrievalResult(
            context=context_text,
            snippets=snippets,
            selected_nodes=selected_nodes,
            selected_edges=selected_edges,
            token_measurement=measurement,
        )

    def _fallback_context(self, repo_id: str) -> GraphRetrievalResult:
        source_dir = self.storage.repo_source_dir(repo_id)
        snippets: list[SourceSnippet] = []
        context_text = ""

        py_files = list(source_dir.rglob("*.py"))
        if py_files:
            chosen = py_files[0]
            text = read_text_lossy(chosen)[:8000]
            context_text = f"Fallback code excerpt from {chosen.name}:\n{text}"
            snippets.append(SourceSnippet(file_path=str(chosen.relative_to(source_dir)), line_start=1, line_end=min(200, text.count('\n') + 1), text=text, source="fallback"))

        measurement = self.token_service.measure_estimated("codegraph_graphify_optimized_context", context_text)
        return GraphRetrievalResult(context=context_text, snippets=snippets, selected_nodes=[], selected_edges=[], token_measurement=measurement)

    def _node_to_snippet(self, node: GraphNode) -> SourceSnippet:
        return SourceSnippet(
            file_path=node.file_path or "unknown",
            line_start=node.line_start or 0,
            line_end=node.line_end or 0,
            text=node.source_snippet or "",
            source="graphify_node",
        )

    def _build_graph_context_text(self, query: str, nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
        lines = [f"Graphify query: {query}", "Selected Graph Nodes:"]
        for node in nodes:
            lines.append("---")
            lines.append(f"Node ID: {node.node_id}")
            lines.append(f"Type: {node.node_type}")
            if node.label:
                lines.append(f"Label: {node.label}")
            if node.file_path:
                lines.append(f"File: {node.file_path}:{node.line_start}-{node.line_end}")
            snippet = node.source_snippet or node.metadata.get("source_snippet") or ""
            if snippet:
                lines.append("Source snippet:")
                lines.append(snippet)
        if edges:
            lines.append("\nSelected Graph Edges:")
            for edge in edges:
                lines.append(f"- {edge.source_node} --{edge.edge_type}--> {edge.target_node}")
        return "\n".join(lines)
